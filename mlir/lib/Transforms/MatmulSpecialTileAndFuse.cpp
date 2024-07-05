//=== MatmulSpecialTileAndFuse.cpp ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass finds two consecutive matmuls,
// tiles them and fuses them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_MATMULSPECIALTILEANDFUSE
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// MatmulSpecialTileAndFuse Pass
//===----------------------------------------------------------------------===//

namespace {
struct MatmulSpecialTileAndFuse
    : public impl::MatmulSpecialTileAndFuseBase<MatmulSpecialTileAndFuse> {
  void runOnOperation() override;
};
} // namespace

void MatmulSpecialTileAndFuse::runOnOperation() {
  Operation *topOp = getOperation();
  MLIRContext *context = topOp->getContext();
  // A ModuleOp contains a single region, which contains a single block.
  auto moduleOp = dyn_cast<ModuleOp>(topOp);
  auto &topFunc =
      topOp->getRegions().front().getBlocks().front().getOperations().front();
  Region &region = topFunc.getRegions().front();
  Block &block = region.getBlocks().front();
  IRRewriter rewriter(context);

  auto mmOps = block.getOps<linalg::MatmulOp>();
  for (linalg::MatmulOp mm : mmOps) {
    llvm::dbgs() << mm << '\n';
    auto tilableOp = dyn_cast<TilingInterface>(mm.getOperation());
    ArrayRef<OpFoldResult> numThreads = {rewriter.getIndexAttr(8)};
    // linalg::tileToForallOp(rewriter, tilableOp, numThreads, std::nullopt);
  }
}

std::unique_ptr<Pass> mlir::createMatmulSpecialTileAndFusePass() {
  return std::make_unique<MatmulSpecialTileAndFuse>();
}