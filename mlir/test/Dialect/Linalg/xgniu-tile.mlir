// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s


// func.func @tile_matmul(%arg0: tensor<64x32xf32>,
//                        %arg1: tensor<32x128xf32>,
//                        %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %0 : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled, %forall = transform.structured.tile_using_forall %0 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     transform.apply_cse to %forall : !transform.any_op
//     transform.yield
//   }
// }

// result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 32)>
// module {
//   func.func @tile_matmul(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %c8 = arith.constant 8 : index
//     %c4 = arith.constant 4 : index
//     %0 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %arg2) -> (tensor<64x128xf32>) {
//       %1 = affine.apply #map(%arg3)
//       %2 = affine.apply #map1(%arg4)
//       %3 = affine.apply #map(%arg3)
//       %4 = affine.apply #map1(%arg4)
//       %5 = affine.apply #map(%arg3)
//       %6 = affine.apply #map1(%arg4)
//       %extracted_slice = tensor.extract_slice %arg0[%3, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg5[%5, %6] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
//       %7 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
//       %8 = affine.apply #map(%arg3)
//       %9 = affine.apply #map1(%arg4)
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %7 into %arg5[%8, %9] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//       }
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }





// func.func @tile_softmax(%arg0: tensor<64x32xf32>,
//                        %arg1: tensor<64x32xf32>
//                        ) -> tensor<64x32xf32> {
//   %0 = tensor.empty() : tensor<64x32xf32>
//   %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<64x32xf32>) outs(%0: tensor<64x32xf32>) -> tensor<64x32xf32>
//   return %1 : tensor<64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled, %forall1 = transform.structured.tile_using_forall %0 tile_sizes [8, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     transform.yield
//   }
// }

// Semantically wrong result. Can not tile reduction axis using tile_using_forall.
// result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 16)>
// module {
//   func.func @tile_softmax(%arg0: tensor<64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<64x32xf32> {
//     %0 = tensor.empty() : tensor<64x32xf32>
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     %c0_0 = arith.constant 0 : index
//     %c1_1 = arith.constant 1 : index
//     %c8 = arith.constant 8 : index
//     %c2 = arith.constant 2 : index
//     %1 = scf.forall (%arg2, %arg3) in (8, 2) shared_outs(%arg4 = %0) -> (tensor<64x32xf32>) {
//       %2 = affine.apply #map(%arg2)
//       %3 = affine.apply #map1(%arg3)
//       %extracted_slice = tensor.extract_slice %arg0[%2, %3] [8, 16] [1, 1] : tensor<64x32xf32> to tensor<8x16xf32>
//       %extracted_slice_2 = tensor.extract_slice %arg4[%2, %3] [8, 16] [1, 1] : tensor<64x32xf32> to tensor<8x16xf32>
//       %4 = linalg.softmax dimension(1) ins(%extracted_slice : tensor<8x16xf32>) outs(%extracted_slice_2 : tensor<8x16xf32>) -> tensor<8x16xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %4 into %arg4[%2, %3] [8, 16] [1, 1] : tensor<8x16xf32> into tensor<64x32xf32>
//       }
//     }
//     return %1 : tensor<64x32xf32>
//   }
// }





// func.func @tile_softmax(%arg0: tensor<64x32xf32>,
//                        %arg1: tensor<64x32xf32>
//                        ) -> tensor<64x32xf32> {
//   %0 = tensor.empty() : tensor<64x32xf32>
//   %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<64x32xf32>) outs(%0: tensor<64x32xf32>) -> tensor<64x32xf32>
//   return %1 : tensor<64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled, %forall1 = transform.structured.tile_using_forall %0 tile_sizes [8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %tiled
//       by num_threads = [0, 8], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//     transform.apply_cse to %loop : !transform.any_op
//     transform.yield
//   }
// }

// Not supported. Softmax is not a PartialReductionOpInterface.





// func.func @tile_softmax_matmul(%arg0: tensor<64x32xf32>,
//                        %arg1: tensor<32x128xf32>,
//                        %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %0 = tensor.empty() : tensor<64x32xf32>
//   %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<64x32xf32>) outs(%0: tensor<64x32xf32>) -> tensor<64x32xf32>
//   %2 = linalg.matmul ins(%1, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %2 : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled, %forall1 = transform.structured.tile_using_forall %1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     %fused, %forall2 = transform.structured.fuse_into_containing_op %0 into %forall1
//         : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
//     transform.yield
//   }
// }





// func.func @matmul_tile_reduction(
//   %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
//                      outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %matmul : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
//       by num_threads = [0, 0, 4], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//     transform.apply_cse to %loop : !transform.any_op
//     transform.yield
//   }
// }

// // result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// module {
//   func.func @matmul_tile_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = tensor.empty() : tensor<64x128x4xf32>
//     %cst = arith.constant 0.000000e+00 : f32
//     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128x4xf32>) -> tensor<64x128x4xf32>
//     %c4 = arith.constant 4 : index
//     %2 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %1) -> (tensor<64x128x4xf32>) {
//       %extracted_slice = tensor.extract_slice %arg4[0, 0, %arg3] [64, 128, 1] [1, 1, 1] : tensor<64x128x4xf32> to tensor<64x128xf32>
//       %4 = affine.apply #map(%arg3)
//       %extracted_slice_0 = tensor.extract_slice %arg0[0, %4] [64, 8] [1, 1] : tensor<64x32xf32> to tensor<64x8xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg1[%4, 0] [8, 128] [1, 1] : tensor<32x128xf32> to tensor<8x128xf32>
//       %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0] [64, 128] [1, 1] : tensor<64x128xf32> to tensor<64x128xf32>
//       %5 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_1 : tensor<64x8xf32>, tensor<8x128xf32>) outs(%extracted_slice_2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %5 into %arg4[0, 0, %arg3] [64, 128, 1] [1, 1, 1] : tensor<64x128xf32> into tensor<64x128x4xf32>
//       }
//     }
//     %3 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<64x128x4xf32>) outs(%arg2 : tensor<64x128xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %4 = arith.addf %in, %out : f32
//       linalg.yield %4 : f32
//     } -> tensor<64x128xf32>
//     return %3 : tensor<64x128xf32>
//   }
// }






func.func @matmul_tile_parallel_reduction(
  %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
                     outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
  return %matmul : tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled_mm, %forall = transform.structured.tile_using_forall %0 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %tiled_mm
      by num_threads = [0, 0, 4], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.apply_cse to %forall : !transform.any_op
    transform.apply_cse to %loop : !transform.any_op
    transform.yield
  }
}

// result:
#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0 * 2)>
module {
  func.func @matmul_tile_parallel_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %0 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %arg2) -> (tensor<64x128xf32>) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map1(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
      %3 = tensor.empty() : tensor<8x32x4xf32>
      %cst = arith.constant 0.000000e+00 : f32
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x32x4xf32>) -> tensor<8x32x4xf32>
      %5 = scf.forall (%arg6) in (4) shared_outs(%arg7 = %4) -> (tensor<8x32x4xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg7[0, 0, %arg6] [8, 32, 1] [1, 1, 1] : tensor<8x32x4xf32> to tensor<8x32xf32>
        %7 = affine.apply #map(%arg6)
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %7] [8, 8] [1, 1] : tensor<8x32xf32> to tensor<8x8xf32>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[%7, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
        %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, 0] [8, 32] [1, 1] : tensor<8x32xf32> to tensor<8x32xf32>
        %8 = linalg.matmul ins(%extracted_slice_3, %extracted_slice_4 : tensor<8x8xf32>, tensor<8x32xf32>) outs(%extracted_slice_5 : tensor<8x32xf32>) -> tensor<8x32xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %8 into %arg7[0, 0, %arg6] [8, 32, 1] [1, 1, 1] : tensor<8x32xf32> into tensor<8x32x4xf32>
        }
      }
      %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<8x32x4xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %out : f32
        linalg.yield %7 : f32
      } -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %6 into %arg5[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
      }
    }
    return %0 : tensor<64x128xf32>
  }
}
