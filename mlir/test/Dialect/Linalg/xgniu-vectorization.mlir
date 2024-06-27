// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

// func.func @vectorize_add(%arg0: tensor<64x32xf32>,
//                          %arg1: tensor<64x32xf32>,
//                          %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
//   %0 = linalg.add ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<64x32xf32>) outs(%arg2 : tensor<64x32xf32>) -> tensor<64x32xf32>
//   return %0 : tensor<64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.generic", "linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     transform.structured.vectorize %0 vector_sizes [64, 32] : !transform.any_op
//     transform.yield
//   }
// }

// result:
// func.func @vectorize_add(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>, %arg2: tensor<64x32xf32>) -> tensor<64x32xf32> {
//   %c64 = arith.constant 64 : index
//   %c32 = arith.constant 32 : index
//   %c0 = arith.constant 0 : index
//   %cst = arith.constant 0.000000e+00 : f32
//   %0 = vector.transfer_read %arg0[%c0, %c0], %cst : tensor<64x32xf32>, vector<64x32xf32>
//   %cst_0 = arith.constant 0.000000e+00 : f32
//   %1 = vector.transfer_read %arg1[%c0, %c0], %cst_0 : tensor<64x32xf32>, vector<64x32xf32>
//   %cst_1 = arith.constant 0.000000e+00 : f32
//   %2 = vector.transfer_read %arg2[%c0, %c0], %cst_1 : tensor<64x32xf32>, vector<64x32xf32>
//   %3 = arith.addf %0, %1 : vector<64x32xf32>
//   %c0_2 = arith.constant 0 : index
//   %4 = vector.transfer_write %3, %arg2[%c0_2, %c0_2] : vector<64x32xf32>, tensor<64x32xf32>
//   return %4 : tensor<64x32xf32>
// }





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
