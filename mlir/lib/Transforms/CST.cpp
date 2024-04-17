//===- CST.cpp - Constant Subgraph Transform -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs a constant subgraph transform in MLIR.
//
//===----------------------------------------------------------------------===//

#include <deque>
#include <unordered_set>

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CST
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// CST Rewrites
//===----------------------------------------------------------------------===//
FailureOr<memref::GlobalOp> createGlobalOp(ModuleOp &moduleOp, Block &block,
                                           Location loc, MemRefType type,
                                           uint64_t alignment,
                                           Attribute memorySpace = {}) {
  // Create a builder without an insertion point. We will insert using the
  // symbol table to guarantee unique names.
  OpBuilder globalBuilder(moduleOp.getContext());
  SymbolTable symbolTable(moduleOp);

  // Add an optional alignment to the global memref.
  IntegerAttr memrefAlignment =
      alignment > 0 ? IntegerAttr::get(globalBuilder.getI64Type(), alignment)
                    : IntegerAttr();
  if (memorySpace)
    type = MemRefType::Builder(type).setMemorySpace(memorySpace);

  auto global = globalBuilder.create<memref::GlobalOp>(
      loc, "__global_",
      /*sym_visibility=*/globalBuilder.getStringAttr("private"),
      /*type=*/type,
      /*initial_value=*/ElementsAttr(),
      /*constant=*/false,
      /*alignment=*/memrefAlignment);
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());

  return global;
}

//===----------------------------------------------------------------------===//
// CST Pass
//===----------------------------------------------------------------------===//

namespace {
struct CST : public impl::CSTBase<CST> {
  void runOnOperation() override;
};
} // namespace

bool isInConstantSubgraph(Operation *op) {
  auto opNamespace = op->getDialect()->getNamespace();
  if (opNamespace == linalg::LinalgDialect::getDialectNamespace() ||
      opNamespace == tensor::TensorDialect::getDialectNamespace() ||
      opNamespace == arith::ArithDialect::getDialectNamespace()) {
    if (op->getAttr("linalg.in_const_subgraph")) {
      return true;
    }
  }
  return false;
}

/*
// Option #1: Operate on memrefs. Create module globals for folded weights and
// first-run flag.
void CST::runOnOperation() {
  Operation *topOp = getOperation();
  IRRewriter rewriter(topOp->getContext());
  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  Block &block = topFunc.getRegions().front().getBlocks().front();
  OpBuilder builder(topOp->getContext());

  // This global indicates if it's the first calling.
  auto loc = block.front().getLoc();
  MemRefType type = MemRefType::Builder({1}, builder.getI1Type());
  FailureOr<memref::GlobalOp> globalOpInit =
      createGlobalOp(moduleOp, block, loc, type, 64);
  if (failed(globalOpInit))
    return;
  memref::GlobalOp &globalMemrefInit = *globalOpInit;
  auto getGlobalOpInit = builder.create<memref::GetGlobalOp>(
      loc, globalMemrefInit.getType(), globalMemrefInit.getName());
  Block::iterator insertPt = Block::iterator(block.front());
  block.getOperations().insert(insertPt++, getGlobalOpInit);
  auto zeroIndexOp = builder.create<arith::ConstantIndexOp>(loc, 0);
  block.getOperations().insert(insertPt++, zeroIndexOp);
  auto initValueLoadOp = builder.create<memref::LoadOp>(
      loc, getGlobalOpInit.getResult(), zeroIndexOp.getResult());
  block.getOperations().insert(insertPt++, initValueLoadOp);

  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isa<memref::AllocOp>(&op)) {
      bool flag = false;
      for (Operation *user : op.getUsers()) {
        if (isInConstantSubgraph(user)) {
          flag = true;
          break;
        }
      }
      if (!flag)
        continue;
      auto type = op.getResult(0).getType();
      FailureOr<memref::GlobalOp> globalOp = createGlobalOp(
          moduleOp, block, op.getLoc(), dyn_cast<MemRefType>(type), 64);
      if (failed(globalOp))
        return;
      memref::GlobalOp &globalMemref = *globalOp;
      auto getGlobal =
          bufferization::replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
              rewriter, &op, globalMemref.getType(), globalMemref.getName());
      // Insert at the back of the block
      // Block::iterator insertPt = Block::iterator(block.end());
      // // Insert before the terminator, if any.
      // if (insertPt == Block::iterator(block.end()) && !block.empty() &&
      //     std::prev(block.end())->hasTrait<OpTrait::IsTerminator>())
      //   insertPt = std::prev(block.end());
      Block::iterator insertPt = Block::iterator(block.front());
      block.getOperations().insert(insertPt, getGlobal);
      llvm::dbgs() << "GetGlobalOp: " << getGlobal << ' '
                   << getGlobal.getResult().getType() << '\n';
    }
  }

  // Construct the block that folds the constant weights and the main block
  Operation *startOp;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isInConstantSubgraph(&op)) {
      startOp = &op;
      break;
    }
  }

  Operation *endOp;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isInConstantSubgraph(&op)) {
      endOp = &op;
    }
  }

  auto constBB = block.splitBlock(startOp->getIterator());
  auto mainBB = constBB->splitBlock(endOp->getNextNode());

  // Set the isInit flag; cf::CondBranchOp to call the two blocks
  builder.setInsertionPointToEnd(constBB);
  Value oneValue = builder.create<arith::ConstantOp>(
      loc, builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), 1));
  auto initValueStoreOp = builder.create<memref::StoreOp>(
      loc, oneValue, getGlobalOpInit.getResult(), zeroIndexOp.getResult());
  builder.create<cf::BranchOp>(constBB->back().getLoc(), mainBB, ValueRange{});

  // func::FuncOp func = cast<func::FuncOp>(topFunc);
  // FunctionType oldType = func.getFunctionType();
  // SmallVector<Type, 4> newInputs(oldType.getInputs());
  // newInputs.push_back(builder.getI1Type());
  // func.setType(
  //     FunctionType::get(oldType.getContext(), newInputs,
  //     oldType.getResults()));
  // auto loc = block.getArguments().back().getLoc();
  // auto newArg = block.addArgument(builder.getI1Type(), loc);
  loc = block.back().getLoc();
  builder.setInsertionPointToEnd(&block);
  Value zeroValue = builder.create<arith::ConstantOp>(
      loc, builder.getI1Type(), builder.getIntegerAttr(builder.getI1Type(), 0));
  auto isInit = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, initValueLoadOp.getResult(), zeroValue);
  builder.create<cf::CondBranchOp>(loc, isInit, constBB,
                                   constBB->getArguments(), mainBB,
                                   mainBB->getArguments());
}
*/

int64_t getTensorSize(TensorType t) {
  Type eleType = t.getElementType();
  unsigned bitWidth = eleType.getIntOrFloatBitWidth() / 8; // bytes
  ArrayRef<int64_t> shape = t.getShape();
  int64_t size = bitWidth;
  for (auto s : shape) {
    size *= s;
  }
  return size;
}

bool canMoveBefore(Operation *op) {
  if (op->getDialect()->getNamespace() ==
      arith::ArithDialect::getDialectNamespace()) {
    return true;
  }

  if (op->getDialect()->getNamespace() !=
      linalg::LinalgDialect::getDialectNamespace()) {
    return false;
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);

  SmallVector<AffineMap> indexingMaps =
      linalgOp.getIndexingMapsArray();
  for (auto &affineMap : indexingMaps) {
    if (!affineMap.isIdentity()) {
      return false;
    }
  }

  SmallVector<utils::IteratorType> iterTypes =
      linalgOp.getIteratorTypesArray();
  for (auto &iterType : iterTypes) {
    if (iterType != utils::IteratorType::parallel) {
      return false;
    }
  }

  if (op->getNumOperands() > 1) {
    int64_t numInputs = linalgOp.getNumDpsInputs();
    int64_t numInits = linalgOp.getNumDpsInits();
    llvm::dbgs() << "NumInputs: " << numInputs << ", NumInits: " << numInits << '\n';
    // definingOp of init should be tensor.empty()
    for (int64_t i = 0; i < numInits; ++i) {
      OpOperand *outOperand = linalgOp.getDpsInitOperand(i);
      auto parentOp = outOperand->get().getDefiningOp();
      if (!isa<tensor::EmptyOp>(parentOp)) {
        return false;
      }
    }
  }

  return true;
}

void postponeBroadcast(Block &block) {
  // auto bcOps = block.getOps<linalg::BroadcastOp>();
  // for (linalg::BroadcastOp bcOp : bcOps) {}
  SmallVector<Operation *> constBcOps;
  for (Operation &op : block.getOperations()) {
    if (isa<linalg::BroadcastOp>(&op)) {
      Operation *bcOp = &op;
      if (isInConstantSubgraph(bcOp)) {
        constBcOps.push_back(bcOp);
      }
    }
  }

  for (auto bcOp : constBcOps) {
    // For topo v -> pack -> bc -> mul -> matmul, we transform
    // it to v -> pack -> mul -> bc -> matmul, so that we can fold
    // v -> pack -> mul. Note that we require the topo to be sequential
    // and all the Values have exactly one user.
    // go upwards to BlockArg
    SmallVector<Operation *> prevOps;
    Operation *currOp = bcOp;
    while (true) {
      if (currOp->getNumOperands() != 1) {
        break;
      }
      Value operand = currOp->getOperand(0);
      if (isa<BlockArgument>(operand)) {
        break;
      } else {
        currOp = operand.getDefiningOp();
        prevOps.push_back(currOp);
      }
    }

    // go downwards to the last constant op
    SmallVector<Operation *> postOps;
    currOp = bcOp;
    while (true) {
      if (currOp->getNumResults() != 1 || !currOp->hasOneUse()) {
        break;
      }
      Value input = currOp->getResult(0);
      currOp = *(input.getUsers().begin());
      Value output = currOp->getResult(0);
      // NOTE: we require that input shape and output shape of curr op to be same. Operations from 
      // tensor dialect, like pack/unpack/concat/collapse_shape/expand_shape/reshape/pad, are not supported.
      // So we simply restrict that currOp to be from arith or linalg.
      if (!isa<TensorType>(input.getType()) ||
          !isa<TensorType>(output.getType()) ||
          dyn_cast<TensorType>(input.getType()).getShape() != dyn_cast<TensorType>(output.getType()).getShape() ||
          !canMoveBefore(currOp)) {
        break;
      }
      if (!isInConstantSubgraph(currOp)) {
        break;
      } else {
        postOps.push_back(currOp);
      }
    }
    llvm::dbgs() << "Post ops size: " << postOps.size() << '\n';
    for (Operation *postOp : postOps) {
      llvm::dbgs() << "Post op:\n" << *postOp << '\n';
    }
    if (postOps.empty()) {
      continue;
    }

    // move bcOp after the last constant op
    auto cloneOptions = Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
    SmallVector<Operation *> newPostOps;
    Value operand = static_cast<Value>(bcOp->getOperand(0));
    ArrayRef<int64_t> shapeBeforeBc = dyn_cast<TensorType>(operand.getType()).getShape();
    size_t postOpId = 0;
    for (Operation *postOp : postOps) {
      // === By cloning original op ===
      // IRMapping mapper;
      // Operation *newPostOp = postOp->clone(mapper, cloneOptions);
      // mapper.map(postOp->getOperand(0), operand);
      // for (auto it : mapper.getValueMap()) {
      //   it.first.replaceAllUsesWith(it.second);
      // }
      // newPostOp->setOperands({operand});
      // for (auto regions : llvm::zip(postOp->getRegions(), newPostOp->getRegions())) {
      //   std::get<0>(regions).cloneInto(&std::get<1>(regions), mapper);
      // }

      // === By create a new op ===
      // Operation::operand_range operands = postOp->getOperands();
      // operands[0] = operand;
      auto oriAttrs = postOp->getAttrDictionary();
      if (!oriAttrs.empty())
        llvm::dbgs() << "Attrs of ori post op: " << *postOp << '\n' << oriAttrs << '\n';

      SmallVector<Type> newOperandTypes;
      for (auto oriType : postOp->getOperandTypes()) {
        TensorType tt = dyn_cast<TensorType>(oriType);
        newOperandTypes.push_back(tt.cloneWith(shapeBeforeBc, tt.getElementType()));
      }
      SmallVector<Type> newResultTypes;
      for (auto oriType : postOp->getResultTypes()) {
        TensorType tt = dyn_cast<TensorType>(oriType);
        newResultTypes.push_back(tt.cloneWith(shapeBeforeBc, tt.getElementType()));
      }
      auto *newPostOp = Operation::create(postOp->getLoc(), postOp->getName(), newResultTypes,
                                      postOp->getOperands(),
                                      /*postOp->getAttrDictionary()*/std::nullopt,
                                      /*postOp->getPropertiesStorage()*/nullptr,
                                      postOp->getSuccessors(), postOp->getNumRegions());
      for (auto [oldRegion, newRegion] : llvm::zip(postOp->getRegions(), newPostOp->getRegions())) {
        newRegion.takeBody(oldRegion);
      }
      // operand.dropAllUses();
      if (postOpId == 0) {
        // Only the first post op needs to replace its operand. Others only needs
        // to call postOp->replaceAllUsesWith(newPostOp->getResults()).
        newPostOp->getOperand(0).replaceAllUsesWith(operand);
      }
      ++postOpId;

      newPostOp->setAttr("linalg.in_const_subgraph", postOp->getAttr("linalg.in_const_subgraph"));
      if (postOp->getDialect()->getNamespace() == linalg::LinalgDialect::getDialectNamespace()) {
        newPostOp->setAttr("operandSegmentSizes", postOp->getAttr("operandSegmentSizes"));

        OpBuilder builder(postOp->getContext());
        size_t indexingMapsSize = dyn_cast<linalg::LinalgOp>(postOp).getIndexingMapsArray().size();
        unsigned rank = shapeBeforeBc.size();
        SmallVector<AffineMap> indexingMaps(indexingMapsSize, builder.getMultiDimIdentityMap(rank));
        auto indexingMapsAttr = builder.getAffineMapArrayAttr(indexingMaps);
        newPostOp->setAttr("indexing_maps", indexingMapsAttr);

        SmallVector<utils::IteratorType> iterTypes = dyn_cast<linalg::LinalgOp>(postOp).getIteratorTypesArray();
        iterTypes.resize(rank);
        auto iterTypesAttr = builder.getArrayAttr(llvm::to_vector(llvm::map_range(
              iterTypes,
              [&](utils::IteratorType iter) -> mlir::Attribute {
                return linalg::IteratorTypeAttr::get(builder.getContext(), iter);
              })));
        newPostOp->setAttr("iterator_types", iterTypesAttr);
      } else {
        // Ops from other dialects.
      }

      // Modify the outputOperands of postOp. Here we simply assume that the value is from tensor.empty().
      if (postOp->getNumOperands() > 0) {
        for (size_t i = 1; i < postOp->getNumOperands(); ++i) {
          auto outOperand = postOp->getOperand(i);
          outOperand.setType(newOperandTypes.front());
        }
      }

      block.getOperations().push_back(newPostOp);
      newPostOp->moveAfter(postOp);
      newPostOps.push_back(newPostOp);
      postOp->replaceAllUsesWith(newPostOp->getResults());

      operand = static_cast<Value>(newPostOp->getResult(0));
      // operand.setType(typeBeforeBc);
    }

    // ERROR: %16 = "linalg.broadcast"(%16, %12)
    // IRMapping mapper;
    // auto newBcOp = bcOp->clone(mapper, cloneOptions);
    // for (auto it : mapper.getValueMap()) {
    //   it.first.replaceAllUsesWith(it.second);
    // }
    // for (auto regions : llvm::zip(bcOp->getRegions(), newBcOp->getRegions())) {
    //   std::get<0>(regions).cloneInto(&std::get<1>(regions), mapper);
    // }
    // newBcOp->setOperands({operand, bcOp->getOperand(1)});

    // ValueRange newBcOperands = {operand, bcOp->getOperand(1)};
    // auto *newBcOp = Operation::create(bcOp->getLoc(), bcOp->getName(), bcOp->getResultTypes(),
    //                                 bcOp->getOperands(), bcOp->getAttrDictionary(),
    //                                 bcOp->getPropertiesStorage(),
    //                                 bcOp->getSuccessors(), bcOp->getNumRegions());
    // for (auto [oldRegion, newRegion] : llvm::zip(bcOp->getRegions(), newBcOp->getRegions())) {
    //   newRegion.takeBody(oldRegion);
    // }

    // auto nextOp = *(newPostOps.back()->getUsers().begin());
    // nextOp->getOperand(0).replaceAllUsesWith(newBcOp->getResult(0));
    // block.getOperations().push_back(newBcOp);
    // newBcOp->moveAfter(newPostOps.back());

    auto nextOp = *(newPostOps.back()->getUsers().begin());
    nextOp->getOperand(0).replaceAllUsesWith(bcOp->getResult(0));
    bcOp->moveAfter(newPostOps.back());
    bcOp->getOperand(0).replaceUsesWithIf(operand, [&](OpOperand &val) {
      Operation *op = val.getOwner();
      return op == bcOp;
    });

    for (auto it = postOps.rbegin(); it != postOps.rend(); ++it) {
      (*it)->erase();
    }
  }

  llvm::dbgs() << "After postponeBroadcast:\n";
  block.print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

// Option #3: Operate on tensors. Create fold() and compute() on module. The
// folded weights and first-run flag is maintained by upper-level runtime.
void CST::runOnOperation() {
  Operation *topOp = getOperation();
  MLIRContext *context = topOp->getContext();
  // A ModuleOp contains a single region, which contains a single block.
  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  SymbolTable symbolTable(moduleOp);
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  OpBuilder builder(context);

  auto topFuncAttr = topFunc.getAttrDictionary();
  std::optional<NamedAttribute> constArgs =
      topFuncAttr.getNamed("linalg.const_args");
  std::unordered_set<int> constArgsIndexes;
  if (constArgs.has_value()) {
    ArrayAttr constArgsArray = llvm::dyn_cast<ArrayAttr>(constArgs->getValue());
    for (auto id : constArgsArray) {
      constArgsIndexes.insert(llvm::cast<IntegerAttr>(id).getInt());
    }
  } else {
    return;
  }
  if (constArgsIndexes.empty()) {
    return;
  }

  Region &region = topFunc.getRegions().front();
  Block &block = region.getBlocks().front();

  postponeBroadcast(block);

  SmallVector<Operation *> constOps;
  for (Operation &op : llvm::make_early_inc_range(block)) {
    if (isInConstantSubgraph(&op)) {
      constOps.push_back(&op);
    }
  }
  // Move the constant ops in front of others.
  // auto prevOp = &(block.getOperations().front());
  // constOps.front()->moveBefore(prevOp);
  // prevOp = constOps.front();
  // for (auto itOp = constOps.begin() + 1; itOp < constOps.end(); ++itOp) {
  //   (*itOp)->moveAfter(prevOp);
  //   prevOp = *itOp;
  // }

  std::string funcName("fold");
  SmallVector<Type> inputTypes; // types of constant weights
  // values of constant weights in original block
  SmallVector<Value> inputValues;
  SmallVector<Type> outputTypes; // types of folded constant weights
  // values of folded constant weights in original block
  SmallVector<Value> outputValues;
  Value v;
  // TODO: solve complicated topology. Currently we only handle simple topology
  // where one constant weight input will and only will produce one constant
  // output and each constant weight only contributes to one constant output.
  for (size_t id = 0; id < block.getNumArguments(); ++id) {
    if (constArgsIndexes.count(id) == 1) {
      auto arg = block.getArgument(id);
      if (!isa<TensorType>(arg.getType())) {
        continue;
      }
      inputTypes.push_back(arg.getType());
      v = dyn_cast<Value>(arg);
      inputValues.push_back(v);
      SmallVector<Value> valuesOnTheWay = {v}; // the constant tensors
      // For v -> pack1 -> pack2 -> matmul, we need the type of output of pack2
      while (!v.getUsers().empty()) {
        // v.getUsers().size() should be 1
        Operation *user = *(v.getUsers().begin());
        if (!isInConstantSubgraph(user)) {
          outputTypes.push_back(v.getType());
          outputValues.push_back(v);
          break;
        }
        // user should has only 1 output value
        OpResult result = *(user->result_begin());
        v = dyn_cast<Value>(result);
        valuesOnTheWay.push_back(v);
      }
      for (auto &v : valuesOnTheWay) {
        llvm::dbgs() << "Value on the way:\n" << v << '\n';
      }
      llvm::dbgs() << '\n';

      // If data size of outputValue is too greater than size of inputValue, do not fold it.
      // Compare data size changes during traverse to find the last op that satisfies this condition.
      int64_t initSize = getTensorSize(dyn_cast<TensorType>(valuesOnTheWay[0].getType()));
      if (!isa<TensorType>(outputTypes.back()) ||
          initSize * 8 < getTensorSize(dyn_cast<TensorType>(outputTypes.back()))) {
        size_t lastIdx = 0;
        for (size_t i = 1; i < valuesOnTheWay.size(); ++i) {
          int64_t size = getTensorSize(dyn_cast<TensorType>(valuesOnTheWay[i].getType()));
          if (initSize * 8 > size) {
            lastIdx = i;
          }
        }
        if (lastIdx == 0) { // no suitable value found
          inputTypes.pop_back();
          outputTypes.pop_back();
          inputValues.pop_back();
          outputValues.pop_back();
          constArgsIndexes.erase(id);
        } else {
          outputTypes.back() = valuesOnTheWay[lastIdx].getType();
          outputValues.back() = valuesOnTheWay[lastIdx];
        }
      }
    }
  }
  if (inputTypes.size() != outputTypes.size()) {
    return;
  }

  FunctionType foldFuncType =
      FunctionType::get(context, inputTypes, outputTypes);
  auto foldFunc =
      builder.create<func::FuncOp>(topFunc.getLoc(), funcName, foldFuncType);
  Block *foldBlock = foldFunc.addEntryBlock();
  // for (Operation *op : constOps) {
  //   // bad strategy: some ops should not move; some ops should be copy
  //   op->moveBefore(foldBlock, foldBlock->end());
  // }
  // values of folded constant weights in foldBlock
  SmallVector<Value> outputValuesInFold;
  IRMapping mapper;
  for (Operation *op : constOps) {
    foldBlock->getOperations().push_back(op->clone(mapper));
  }
  // the order of outputValuesInFold is according to the order of corresponding
  // inputValues
  for (auto &v : outputValues) {
    auto foldedV = mapper.lookupOrNull(v);
    outputValuesInFold.push_back(foldedV);
    v.replaceUsesWithIf(foldedV, [&](OpOperand &val) {
      Operation *op = val.getOwner();
      return op->getBlock() == foldBlock;
    });
  }

  auto returnOp =
      builder.create<func::ReturnOp>(topOp->getLoc(), outputValuesInFold);
  foldBlock->getOperations().push_back(returnOp);
  for (size_t i = 0; i < inputValues.size(); ++i) {
    inputValues[i].replaceUsesWithIf(foldBlock->getArgument(i),
                                     [&](OpOperand &val) {
                                       Operation *op = val.getOwner();
                                       return op->getBlock() == foldBlock;
                                     });
  }

  foldFunc.setVisibility(SymbolTable::Visibility::Public);
  moduleOp.push_back(foldFunc);
  symbolTable.insert(foldFunc);

  // modify the BlockArguments of block
  size_t oriNumArgs = block.getNumArguments();
  size_t argIdx = 0;
  /* Option 1: Add folded args to end; delete dead ops; erase original args.
  for (size_t id = 0; id < oriNumArgs; ++id) {
    if (constArgsIndexes.count(id) == 1) {
      // add the folded argument at the end
      auto loc = block.getArgument(id).getLoc();

      BlockArgument foldArg = block.insertArgument(block.getNumArguments(),
                                                   outputTypes[argIdx], loc);
      outputValues[argIdx].replaceUsesWithIf(foldArg, [&](OpOperand &val) {
        Operation *op = val.getOwner();
        return op->getBlock() == &block;
      });
      ++argIdx;
    }
  }

  // Need to delete useless operations first

  // erase unused arguments
  llvm::BitVector argsToErase;
  for (size_t id = 0; id < block.getNumArguments(); ++id) {
    if (constArgsIndexes.count(id) == 1) {
      argsToErase.push_back(true);
    } else {
      argsToErase.push_back(false);
    }
  }
  block.eraseArguments(argsToErase);
  */

  // Option 2
  for (size_t id = 0; id < oriNumArgs; ++id) {
    if (constArgsIndexes.count(id) == 1) {
      auto loc = block.getArgument(id).getLoc();
      BlockArgument foldArg =
          block.insertArgument(id, outputTypes[argIdx], loc);
      outputValues[argIdx].replaceUsesWithIf(foldArg, [&](OpOperand &val) {
        Operation *op = val.getOwner();
        return op->getBlock() == &block;
      });

      std::deque<Value> dq;
      SmallVector<Operation *> opsToErase;
      dq.push_back(block.getArgument(id + 1));
      while (!dq.empty()) {
        Value v = dq.front();
        dq.pop_front();
        for (Operation *op : v.getUsers()) {
          for (auto res : op->getResults()) {
            dq.push_back(res);
          }
          opsToErase.push_back(op);
        }
      }

      for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it) {
        (*it)->erase();
      }
      block.eraseArgument(id + 1);
      ++argIdx;
    }
  }

  // modify the compute func signature
  func::FuncOp computeFunc = cast<func::FuncOp>(topFunc);
  FunctionType computeFuncType = computeFunc.getFunctionType();
  computeFunc.setType(FunctionType::get(context, block.getArgumentTypes(),
                                        computeFuncType.getResults()));

  // Delete dead operations by dialects' canonicalizer
  RewritePatternSet owningPatterns(context);
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(owningPatterns);

  ArrayRef<std::string> disabledPatterns, enabledPatterns;
  std::shared_ptr<const FrozenRewritePatternSet> patterns =
      std::make_shared<FrozenRewritePatternSet>(
          std::move(owningPatterns), disabledPatterns, enabledPatterns);
  GreedyRewriteConfig config;
  LogicalResult converged =
      applyPatternsAndFoldGreedily(topOp, *patterns, config);

  // clean up the constant-related attrs on ops
  for (auto &op : block.getOperations()) {
    if (op.getAttr("linalg.in_const_subgraph")) {
      op.removeAttr("linalg.in_const_subgraph");
    }
  }
  for (auto &op : foldBlock->getOperations()) {
    if (op.getAttr("linalg.in_const_subgraph")) {
      op.removeAttr("linalg.in_const_subgraph");
    }
  }
}

std::unique_ptr<Pass> mlir::createCSTPass() { return std::make_unique<CST>(); }
