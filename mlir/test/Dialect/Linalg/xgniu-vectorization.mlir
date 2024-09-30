// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// func.func @tile_vectorize_add(%arg0: tensor<16x64x32xf32>,
//                          %arg1: tensor<16x64x32xf32>,
//                          %arg2: tensor<16x64x32xf32>) -> tensor<16x64x32xf32> {
//   %0 = linalg.add ins(%arg0, %arg1 : tensor<16x64x32xf32>, tensor<16x64x32xf32>) outs(%arg2 : tensor<16x64x32xf32>) -> tensor<16x64x32xf32>
//   return %0 : tensor<16x64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.generic", "linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op

//     %tiled, %forall = transform.structured.tile_using_forall %0 tile_sizes [4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     transform.structured.vectorize %tiled vector_sizes [4, 8, 32] : !transform.any_op

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// // result:
// #map = affine_map<(d0) -> (d0 * 4)>
// #map1 = affine_map<(d0) -> (d0 * 8)>
// module {
//   func.func @tile_vectorize_add(%arg0: tensor<16x64x32xf32>, %arg1: tensor<16x64x32xf32>, %arg2: tensor<16x64x32xf32>) -> tensor<16x64x32xf32> {
//     %0 = scf.forall (%arg3, %arg4) in (4, 8) shared_outs(%arg5 = %arg2) -> (tensor<16x64x32xf32>) {
//       %1 = affine.apply #map(%arg3)
//       %2 = affine.apply #map1(%arg4)
//       %extracted_slice = tensor.extract_slice %arg0[%1, %2, 0] [4, 8, 32] [1, 1, 1] : tensor<16x64x32xf32> to tensor<4x8x32xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[%1, %2, 0] [4, 8, 32] [1, 1, 1] : tensor<16x64x32xf32> to tensor<4x8x32xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2, 0] [4, 8, 32] [1, 1, 1] : tensor<16x64x32xf32> to tensor<4x8x32xf32>
//       %c0 = arith.constant 0 : index
//       %cst = arith.constant 0.000000e+00 : f32
//       %3 = vector.transfer_read %extracted_slice[%c0, %c0, %c0], %cst : tensor<4x8x32xf32>, vector<4x8x32xf32>
//       %4 = vector.transfer_read %extracted_slice_0[%c0, %c0, %c0], %cst : tensor<4x8x32xf32>, vector<4x8x32xf32>
//       %5 = arith.addf %3, %4 : vector<4x8x32xf32>
//       %6 = vector.transfer_write %5, %extracted_slice_1[%c0, %c0, %c0] : vector<4x8x32xf32>, tensor<4x8x32xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %6 into %arg5[%1, %2, 0] [4, 8, 32] [1, 1, 1] : tensor<4x8x32xf32> into tensor<16x64x32xf32>
//       }
//     }
//     return %0 : tensor<16x64x32xf32>
//   }
// }



// func.func @pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
//   %pack = tensor.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x8x16xf32> -> tensor<4x1x32x16x2xf32>
//   return %pack : tensor<4x1x32x16x2xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
//     // For tensor.pack op, the vector_sizes is the outer dims after packing and can be ommitted.
//     transform.structured.vectorize %0 vector_sizes [4, 1, 32] : !transform.any_op
//     transform.yield 
//   }
// }

// result:
// module {
//   func.func @pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
//     %cst = arith.constant 0.000000e+00 : f32
//     %c0 = arith.constant 0 : index
//     %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<32x8x16xf32>, vector<32x8x16xf32>
//     %1 = vector.shape_cast %0 : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
//     %2 = vector.transpose %1, [1, 3, 0, 4, 2] : vector<32x4x2x1x16xf32> to vector<4x1x32x16x2xf32>
//     %3 = tensor.empty() : tensor<4x1x32x16x2xf32>
//     %c0_0 = arith.constant 0 : index
//     %4 = vector.transfer_write %2, %3[%c0_0, %c0_0, %c0_0, %c0_0, %c0_0] {in_bounds = [true, true, true, true, true]} : vector<4x1x32x16x2xf32>, tensor<4x1x32x16x2xf32>
//     return %4 : tensor<4x1x32x16x2xf32>
//   }
// }





func.func @pack_add(%arg0: tensor<4x16x16xf32>, %arg1: tensor<4x16x16xf32>) -> tensor<4x4x4x4x4xf32> {
  %0 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %1 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %2 = tensor.pack %arg0 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %0 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %3 = tensor.pack %arg1 outer_dims_perm = [1, 0, 2] inner_dims_pos = [1, 2] inner_tiles = [4, 4] into %1 : tensor<4x16x16xf32> -> tensor<4x4x4x4x4xf32>
  %4 = tensor.empty() : tensor<4x4x4x4x4xf32>
  %6 = linalg.add ins(%2, %3 : tensor<4x4x4x4x4xf32>, tensor<4x4x4x4x4xf32>) outs(%4: tensor<4x4x4x4x4xf32>) -> tensor<4x4x4x4x4xf32>
  return %6 : tensor<4x4x4x4x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    %tiled, %forall = transform.structured.tile_using_forall %0 tile_sizes [4, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %tiled: !transform.any_op

    %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op
    transform.yield
  }
}





// func.func @vectorize_generic_with_mask(%arg0: tensor<64x32xf32>,
//                                        %arg1: tensor<64x32xf32>,
//                                        %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
//   %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//                                          affine_map<(d0, d1) -> (d0, d1)>,
//                                          affine_map<(d0, d1) -> (d0, d1)>],
//                    iterator_types = ["parallel", "parallel"] }
//     ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<64x32xf32>)
//     outs(%arg2 : tensor<64x32xf32>) {
//     ^bb(%in0: f32, %in1: f32, %out: f32) :
//       %0 = arith.addf %in0, %in1 : f32
//       linalg.yield %0 : f32
//     } -> tensor<64x32xf32>
//   return %0 : tensor<64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     transform.structured.vectorize %0 vector_sizes [64, 64] : !transform.any_op
//     transform.yield
//   }
// }

// result:
// func.func @vectorize_generic_with_mask(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
//   %c64 = arith.constant 64 : index
//   %c32 = arith.constant 32 : index
//   %c0 = arith.constant 0 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %0 = vector.create_mask %c64, %c32 : vector<64x64xi1>
//   %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<64x32xf32>, vector<64x64xf32> } : vector<64x64xi1> -> vector<64x64xf32>
//   %cst_0 = arith.constant 0.000000e+00 : f32
//   %2 = vector.mask %0 { vector.transfer_read %arg1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<64x32xf32>, vector<64x64xf32> } : vector<64x64xi1> -> vector<64x64xf32>
//   %cst_1 = arith.constant 0.000000e+00 : f32
//   %3 = vector.mask %0 { vector.transfer_read %arg2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : tensor<64x32xf32>, vector<64x64xf32> } : vector<64x64xi1> -> vector<64x64xf32>
//   %4 = arith.addf %1, %2 : vector<64x64xf32>
//   %c0_2 = arith.constant 0 : index
//   %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0_2, %c0_2] {in_bounds = [true, true]} : vector<64x64xf32>, tensor<64x32xf32> } : vector<64x64xi1> -> tensor<64x32xf32>
//   return %5 : tensor<64x32xf32>
// }





// func.func @vectorize_matmul(%arg0: tensor<64x32xf32>,
//                             %arg1: tensor<32x128xf32>,
//                             %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %0 : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.generic", "linalg.add", "linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     transform.structured.vectorize %0 vector_sizes [64, 128, 32] : !transform.any_op
//     transform.yield
//   }
// }

// result:
// func.func @vectorize_matmul(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %c64 = arith.constant 64 : index
//   %c128 = arith.constant 128 : index
//   %c32 = arith.constant 32 : index
//   %c0 = arith.constant 0 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %0 = vector.transfer_read %arg0[%c0, %c0], %cst {permutation_map = #map} : tensor<64x32xf32>, vector<64x128x32xf32>
//   %cst_0 = arith.constant 0.000000e+00 : f32
//   %1 = vector.transfer_read %arg1[%c0, %c0], %cst_0 {permutation_map = #map1} : tensor<32x128xf32>, vector<64x128x32xf32>
//   %cst_1 = arith.constant 0.000000e+00 : f32
//   %2 = vector.transfer_read %arg2[%c0, %c0], %cst_1 : tensor<64x128xf32>, vector<64x128xf32>
//   %3 = arith.mulf %0, %1 : vector<64x128x32xf32>
//   %4 = vector.multi_reduction <add>, %3, %2 [2] : vector<64x128x32xf32> to vector<64x128xf32>
//   %c0_2 = arith.constant 0 : index
//   %5 = vector.transfer_write %4, %arg2[%c0_2, %c0_2] : vector<64x128xf32>, tensor<64x128xf32>
//   return %5 : tensor<64x128xf32>
// }
