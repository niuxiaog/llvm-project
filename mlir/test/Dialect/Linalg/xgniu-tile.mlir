// RUN: mlir-opt -transform-interpreter -split-input-file ./mlir/test/Dialect/Linalg/xgniu-tile.mlir


// func.func @tile_matmul(%arg0: tensor<64x32xf32>,
//                        %arg1: tensor<32x128xf32>,
//                        %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %0 : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     // %tiled, %forall = transform.structured.tile_using_forall %0 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     // transform.apply_cse to %forall : !transform.any_op
//     %tiled, %loops:2 = transform.structured.tile_using_for %0 tile_sizes [8, 32, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// result using forall:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 32)>
// module {
//   func.func @tile_matmul(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %arg2) -> (tensor<64x128xf32>) {
//       %1 = affine.apply #map(%arg3)
//       %2 = affine.apply #map1(%arg4)
//       %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
//       %3 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %3 into %arg5[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//       }
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }

// result using for:
// module {
//   func.func @tile_matmul(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %c0 = arith.constant 0 : index
//     %c64 = arith.constant 64 : index
//     %c128 = arith.constant 128 : index
//     %c8 = arith.constant 8 : index
//     %c32 = arith.constant 32 : index
//     %0 = scf.for %arg3 = %c0 to %c64 step %c8 iter_args(%arg4 = %arg2) -> (tensor<64x128xf32>) {
//       %1 = scf.for %arg5 = %c0 to %c128 step %c32 iter_args(%arg6 = %arg4) -> (tensor<64x128xf32>) {
//         %extracted_slice = tensor.extract_slice %arg0[%arg3, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//         %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg5] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
//         %extracted_slice_1 = tensor.extract_slice %arg6[%arg3, %arg5] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
//         %2 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
//         %inserted_slice = tensor.insert_slice %2 into %arg6[%arg3, %arg5] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//         scf.yield %inserted_slice : tensor<64x128xf32>
//       }
//       scf.yield %1 : tensor<64x128xf32>
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }





// func.func @tile_softmax(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
//   %0 = tensor.empty() : tensor<64x32xf32>
//   %1 = linalg.softmax dimension(1) ins(%arg0 : tensor<64x32xf32>) outs(%0: tensor<64x32xf32>) -> tensor<64x32xf32>
//   return %1 : tensor<64x32xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.softmax"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled, %forall1 = transform.structured.tile_using_forall %0 tile_sizes [8, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// Semantically wrong result. Can not tile reduction axis using tile_using_forall.
// result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 16)>
// module {
//   func.func @tile_softmax(%arg0: tensor<64x32xf32>) -> tensor<64x32xf32> {
//     %0 = tensor.empty() : tensor<64x32xf32>
//     %1 = scf.forall (%arg1, %arg2) in (8, 2) shared_outs(%arg3 = %0) -> (tensor<64x32xf32>) {
//       %2 = affine.apply #map(%arg1)
//       %3 = affine.apply #map1(%arg2)
//       %extracted_slice = tensor.extract_slice %arg0[%2, %3] [8, 16] [1, 1] : tensor<64x32xf32> to tensor<8x16xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg3[%2, %3] [8, 16] [1, 1] : tensor<64x32xf32> to tensor<8x16xf32>
//       %4 = linalg.softmax dimension(1) ins(%extracted_slice : tensor<8x16xf32>) outs(%extracted_slice_0 : tensor<8x16xf32>) -> tensor<8x16xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %4 into %arg3[%2, %3] [8, 16] [1, 1] : tensor<8x16xf32> into tensor<64x32xf32>
//       }
//     }
//     return %1 : tensor<64x32xf32>
//   }
// }





// tile_reduction_using_forall: Tile a PartialReductionOpInterface op to a tiled scf.forall doing partial reduction.
// This transformation tiles the target along the reduction dimensions. It creates a tensor initialized with the identity value. Then it creates a scf.forall loops with the number threads given by num_threads. The op is tiled with a size equal to floordiv(size, num_threads). All the partial reduction values are parallel inserted to the new created tensor. After the loop, a merge operation is created to do the final reduction with the partial reductions tensor. If an extra tile_sizes parameter is passed, the tiles are cyclically distributed on the threads of the scf.foralls loop.

// func.func @matmul_tile_reduction(
//   %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
//                           outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %matmul : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
//       by num_threads = [0, 0, 4], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//     // Wrong: %tiled, %forall = transform.structured.tile_using_forall %0 num_threads [0, 0, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// // result:
// #map = affine_map<(d0) -> (d0 * 8)>
// module {
//   func.func @matmul_tile_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = tensor.empty() : tensor<64x128x4xf32>
//     %cst = arith.constant 0.000000e+00 : f32
//     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128x4xf32>) -> tensor<64x128x4xf32>
//     %2 = scf.forall (%arg3) in (4) shared_outs(%arg4 = %1) -> (tensor<64x128x4xf32>) {
//       %extracted_slice = tensor.extract_slice %arg4[0, 0, %arg3] [64, 128, 1] [1, 1, 1] : tensor<64x128x4xf32> to tensor<64x128xf32>
//       %3 = affine.apply #map(%arg3)
//       %extracted_slice_0 = tensor.extract_slice %arg0[0, %3] [64, 8] [1, 1] : tensor<64x32xf32> to tensor<64x8xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg1[%3, 0] [8, 128] [1, 1] : tensor<32x128xf32> to tensor<8x128xf32>
//       %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, 0] [64, 128] [1, 1] : tensor<64x128xf32> to tensor<64x128xf32>
//       %4 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_1 : tensor<64x8xf32>, tensor<8x128xf32>) outs(%extracted_slice_2 : tensor<64x128xf32>) -> tensor<64x128xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %4 into %arg4[0, 0, %arg3] [64, 128, 1] [1, 1, 1] : tensor<64x128xf32> into tensor<64x128x4xf32>
//       }
//     }
//     %reduced = linalg.reduce ins(%2 : tensor<64x128x4xf32>) outs(%arg2 : tensor<64x128xf32>) dimensions = [2]
//       (%in: f32, %init: f32) {
//         %3 = arith.addf %in, %init : f32
//         linalg.yield %3 : f32
//       }
//     return %reduced : tensor<64x128xf32>
//   }
// }






// func.func @matmul_tile_reduction(
//   %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
//                           outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %matmul : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
//       by tile_sizes = [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//     // Wrong(?): %tiled, %forall = transform.structured.tile_using_for %0 tile_sizes [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// result:
// #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// module {
//   func.func @matmul_tile_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = tensor.empty() : tensor<64x128x8xf32>
//     %cst = arith.constant 0.000000e+00 : f32
//     %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128x8xf32>) -> tensor<64x128x8xf32>
//     %c0 = arith.constant 0 : index
//     %c32 = arith.constant 32 : index
//     %c8 = arith.constant 8 : index
//     %2 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %1) -> (tensor<64x128x8xf32>) {
//       %extracted_slice = tensor.extract_slice %arg0[0, %arg3] [64, 8] [1, 1] : tensor<64x32xf32> to tensor<64x8xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [8, 128] [1, 1] : tensor<32x128xf32> to tensor<8x128xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg4[0, 0, 0] [64, 128, 8] [1, 1, 1] : tensor<64x128x8xf32> to tensor<64x128x8xf32>
//       %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<64x8xf32>, tensor<8x128xf32>) outs(%extracted_slice_1 : tensor<64x128x8xf32>) {
//       ^bb0(%in: f32, %in_2: f32, %out: f32):
//         %4 = arith.mulf %in, %in_2 : f32
//         %5 = arith.addf %out, %4 : f32
//         linalg.yield %5 : f32
//       } -> tensor<64x128x8xf32>
//       %inserted_slice = tensor.insert_slice %3 into %arg4[0, 0, 0] [64, 128, 8] [1, 1, 1] : tensor<64x128x8xf32> into tensor<64x128x8xf32>
//       scf.yield %inserted_slice : tensor<64x128x8xf32>
//     }
//     %reduced = linalg.reduce ins(%2 : tensor<64x128x8xf32>) outs(%arg2 : tensor<64x128xf32>) dimensions = [2]
//       (%in: f32, %init: f32) {
//         %3 = arith.addf %in, %init : f32
//         linalg.yield %3 : f32
//       }
//     return %reduced : tensor<64x128xf32>
//   }
// }



func.func @add_tile_reduction(%input : tensor<16x32x64xf32>) -> tensor<16x64xf32> {
  %init = tensor.empty() : tensor<16x64xf32>
  %reduced = linalg.reduce ins(%input : tensor<16x32x64xf32>) outs(%init : tensor<16x64xf32>) dimensions = [1]
    (%in: f32, %out: f32) {
      %0 = arith.addf %out, %in: f32
      linalg.yield %0: f32
    }
  return %reduced : tensor<16x64xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
      by tile_sizes = [0, 16, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %0
    //   by num_threads = [0, 2, 0], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op
    transform.yield
  }
}

// result:
// #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// module {
//   func.func @add_tile_reduction(%arg0: tensor<16x32x64xf32>) -> tensor<16x64xf32> {
//     %0 = tensor.empty() : tensor<16x64xf32>
//     %1 = tensor.empty() : tensor<16x16x64xf32>
//     %cst = arith.constant 0.000000e+00 : f32
//     %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<16x16x64xf32>) -> tensor<16x16x64xf32>
//     %c0 = arith.constant 0 : index
//     %c32 = arith.constant 32 : index
//     %c16 = arith.constant 16 : index
//     %3 = scf.for %arg1 = %c0 to %c32 step %c16 iter_args(%arg2 = %2) -> (tensor<16x16x64xf32>) {
//       %extracted_slice = tensor.extract_slice %arg0[0, %arg1, 0] [16, 16, 64] [1, 1, 1] : tensor<16x32x64xf32> to tensor<16x16x64xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg2[0, 0, 0] [16, 64, 16] [1, 1, 1] : tensor<16x16x64xf32> to tensor<16x64x16xf32>
//       %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<16x16x64xf32>) outs(%extracted_slice_0 : tensor<16x64x16xf32>) {
//       ^bb0(%in: f32, %out: f32):
//         %5 = arith.addf %out, %in : f32
//         linalg.yield %5 : f32
//       } -> tensor<16x64x16xf32>
//       %inserted_slice = tensor.insert_slice %4 into %arg2[0, 0, 0] [16, 64, 16] [1, 1, 1] : tensor<16x64x16xf32> into tensor<16x16x64xf32>
//       scf.yield %inserted_slice : tensor<16x16x64xf32>
//     }
//     %reduced = linalg.reduce ins(%3 : tensor<16x16x64xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [1]
//       (%in: f32, %init: f32) {
//         %4 = arith.addf %in, %init : f32
//         linalg.yield %4 : f32
//       }
//     return %reduced : tensor<16x64xf32>
//   }
// }





// func.func @matmul_tile_parallel_reduction(
//   %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
//                      outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %matmul : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled_mm, %forall = transform.structured.tile_using_forall %0 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %tiled_mm
//       by num_threads = [0, 0, 4], tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// // result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 32)>
// module {
//   func.func @matmul_tile_parallel_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %arg2) -> (tensor<64x128xf32>) {
//       %1 = affine.apply #map(%arg3)
//       %2 = affine.apply #map1(%arg4)
//       %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>

//       %3 = tensor.empty() : tensor<8x32x4xf32>
//       %cst = arith.constant 0.000000e+00 : f32
//       %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x32x4xf32>) -> tensor<8x32x4xf32>
//       %5 = scf.forall (%arg6) in (4) shared_outs(%arg7 = %4) -> (tensor<8x32x4xf32>) {
//         %extracted_slice_2 = tensor.extract_slice %arg7[0, 0, %arg6] [8, 32, 1] [1, 1, 1] : tensor<8x32x4xf32> to tensor<8x32xf32>
//         %6 = affine.apply #map(%arg6)
//         %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %6] [8, 8] [1, 1] : tensor<8x32xf32> to tensor<8x8xf32>
//         %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[%6, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
//         %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, 0] [8, 32] [1, 1] : tensor<8x32xf32> to tensor<8x32xf32>
//         %7 = linalg.matmul ins(%extracted_slice_3, %extracted_slice_4 : tensor<8x8xf32>, tensor<8x32xf32>) outs(%extracted_slice_5 : tensor<8x32xf32>) -> tensor<8x32xf32>
//         scf.forall.in_parallel {
//           tensor.parallel_insert_slice %7 into %arg7[0, 0, %arg6] [8, 32, 1] [1, 1, 1] : tensor<8x32xf32> into tensor<8x32x4xf32>
//         }
//       }
//       %reduced = linalg.reduce ins(%5 : tensor<8x32x4xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) dimensions = [2]
//         (%in: f32, %init: f32) {
//           %6 = arith.addf %in, %init : f32
//           linalg.yield %6 : f32
//         }

//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %reduced into %arg5[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//       }
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }






// func.func @matmul_tile_parallel_reduction(
//   %A: tensor<64x32xf32>, %B: tensor<32x128xf32>, %out: tensor<64x128xf32>) -> tensor<64x128xf32> {
//   %matmul = linalg.matmul ins(%A, %B: tensor<64x32xf32>, tensor<32x128xf32>)
//                      outs(%out: tensor<64x128xf32>) -> tensor<64x128xf32>
//   return %matmul : tensor<64x128xf32>
// }

// module attributes {transform.with_named_sequence} {
//   transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//     %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
//     %tiled_mm, %forall = transform.structured.tile_using_forall %0 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %tiled_mm
//       by tile_sizes = [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

//     %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
//     transform.apply_cse to %func_op : !transform.any_op
//     transform.apply_dce to %func_op : !transform.any_op
//     transform.yield
//   }
// }

// // rsult:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 32)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// module {
//   func.func @matmul_tile_parallel_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %0 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %arg2) -> (tensor<64x128xf32>) {
//       %1 = affine.apply #map(%arg3)
//       %2 = affine.apply #map1(%arg4)
//       %extracted_slice = tensor.extract_slice %arg0[%1, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//       %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
//       %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>

//       %3 = tensor.empty() : tensor<8x32x8xf32>
//       %cst = arith.constant 0.000000e+00 : f32
//       %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8x32x8xf32>) -> tensor<8x32x8xf32>
//       %c0 = arith.constant 0 : index
//       %c32 = arith.constant 32 : index
//       %c8 = arith.constant 8 : index
//       %5 = scf.for %arg6 = %c0 to %c32 step %c8 iter_args(%arg7 = %4) -> (tensor<8x32x8xf32>) {
//         %extracted_slice_2 = tensor.extract_slice %extracted_slice[0, %arg6] [8, 8] [1, 1] : tensor<8x32xf32> to tensor<8x8xf32>
//         %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
//         %extracted_slice_4 = tensor.extract_slice %arg7[0, 0, 0] [8, 32, 8] [1, 1, 1] : tensor<8x32x8xf32> to tensor<8x32x8xf32>
//         %6 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<8x8xf32>, tensor<8x32xf32>) outs(%extracted_slice_4 : tensor<8x32x8xf32>) {
//         ^bb0(%in: f32, %in_5: f32, %out: f32):
//           %7 = arith.mulf %in, %in_5 : f32
//           %8 = arith.addf %out, %7 : f32
//           linalg.yield %8 : f32
//         } -> tensor<8x32x8xf32>
//         %inserted_slice = tensor.insert_slice %6 into %arg7[0, 0, 0] [8, 32, 8] [1, 1, 1] : tensor<8x32x8xf32> into tensor<8x32x8xf32>
//         scf.yield %inserted_slice : tensor<8x32x8xf32>
//       }
//       %reduced = linalg.reduce ins(%5 : tensor<8x32x8xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) dimensions = [2]
//         (%in: f32, %init: f32) {
//           %6 = arith.addf %in, %init : f32
//           linalg.yield %6 : f32
//         }

//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %reduced into %arg5[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//       }
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }
