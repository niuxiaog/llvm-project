// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s


func.func @tile_matmul(%arg0: tensor<64x32xf32>,
                       %arg1: tensor<32x128xf32>,
                       %arg2: tensor<64x128xf32>,
                       %arg3: tensor<128x16xf32>,
                       %arg4: tensor<64x16xf32>) -> tensor<64x16xf32> attributes {llvm.emit_c_interface} {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %3 = tensor.empty() : tensor<64x128xf32>
  %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %5 = linalg.add ins(%2, %arg2 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%4 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %6 = tensor.empty() : tensor<64x128xf32>
  %7 = linalg.fill ins(%cst_0 : f32) outs(%6 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %8 = linalg.generic { indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>], iterator_types = ["parallel", "parallel"] } ins(%5 : tensor<64x128xf32>) outs(%7 : tensor<64x128xf32>) {
  ^bb0(%in_one : f32, %out_one : f32):
    %c0 = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %in_one, %c0 : f32
    %sel = arith.select %cmp, %in_one, %c0 : f32
    linalg.yield %sel : f32 
  } -> tensor<64x128xf32>
  %9 = tensor.empty() : tensor<64x16xf32>
  %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<64x16xf32>) -> tensor<64x16xf32>
  %11 = linalg.matmul {matmul_special_tiling = true} ins(%8, %arg3 : tensor<64x128xf32>, tensor<128x16xf32>) outs(%10 : tensor<64x16xf32>) -> tensor<64x16xf32>
  %12 = tensor.empty() : tensor<64x16xf32>
  %13 = linalg.fill ins(%cst_0 : f32) outs(%12 : tensor<64x16xf32>) -> tensor<64x16xf32>
  %14 = linalg.sub ins(%11, %arg4 : tensor<64x16xf32>, tensor<64x16xf32>) outs(%13 : tensor<64x16xf32>) -> tensor<64x16xf32>
  return %14 : tensor<64x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %mm0, %mm1 = transform.split_handle %1: (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %2 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    %tiled, %forall1 = transform.structured.tile_using_forall %mm1 tile_sizes [8, 0, 10]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused2, %forall2 = transform.structured.fuse_into_containing_op %2 into %forall1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused3, %forall3 = transform.structured.fuse_into_containing_op %0 into %forall2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused4, %forall4 = transform.structured.fuse_into_containing_op %mm0 into %forall3
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %func_op = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op
    transform.yield
  }
}

// result with new tiling:
// #map = affine_map<(d0) -> (d0 * 8)>
// module {
//   func.func @tile_matmul(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>, %arg3: tensor<128x16xf32>, %arg4: tensor<64x16xf32>, %arg5: tensor<64x16xf32>) -> tensor<64x16xf32> {
//     %0 = tensor.empty() : tensor<64x128xf32>
//     %1 = tensor.empty() : tensor<64x16xf32>
//     %2 = scf.forall (%arg6) in (8) shared_outs(%arg7 = %1) -> (tensor<64x16xf32>) {
//       %4 = affine.apply #map(%arg6)
//       %extracted_slice = tensor.extract_slice %arg7[%4, 0] [8, 16] [1, 1] : tensor<64x16xf32> to tensor<8x16xf32>
//       %c0 = arith.constant 0 : index
//       %c16 = arith.constant 16 : index
//       %c1 = arith.constant 1 : index
//       %5 = scf.for %arg8 = %c0 to %c16 step %c1 iter_args(%arg9 = %extracted_slice) -> (tensor<8x16xf32>) {
//         %6 = affine.apply #map(%arg8)
//         %extracted_slice_0 = tensor.extract_slice %arg0[%4, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
//         %extracted_slice_1 = tensor.extract_slice %arg1[0, %6] [32, 8] [1, 1] : tensor<32x128xf32> to tensor<32x8xf32>
//         %extracted_slice_2 = tensor.extract_slice %0[%4, %6] [8, 8] [1, 1] : tensor<64x128xf32> to tensor<8x8xf32>
//         %7 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_1 : tensor<8x32xf32>, tensor<32x8xf32>) outs(%extracted_slice_2 : tensor<8x8xf32>) -> tensor<8x8xf32>
//         %extracted_slice_3 = tensor.extract_slice %arg2[%4, %6] [8, 8] [1, 1] : tensor<64x128xf32> to tensor<8x8xf32>
//         %8 = linalg.add ins(%7, %extracted_slice_3 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%extracted_slice_2 : tensor<8x8xf32>) -> tensor<8x8xf32>
//         %extracted_slice_4 = tensor.extract_slice %arg3[%6, 0] [8, 16] [1, 1] : tensor<128x16xf32> to tensor<8x16xf32>
//         %9 = linalg.matmul ins(%8, %extracted_slice_4 : tensor<8x8xf32>, tensor<8x16xf32>) outs(%arg9 : tensor<8x16xf32>) -> tensor<8x16xf32>
//         scf.yield %9 : tensor<8x16xf32>
//       }
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %5 into %arg7[%4, 0] [8, 16] [1, 1] : tensor<8x16xf32> into tensor<64x16xf32>
//       }
//     }
//     %3 = linalg.sub ins(%2, %arg4 : tensor<64x16xf32>, tensor<64x16xf32>) outs(%arg5 : tensor<64x16xf32>) -> tensor<64x16xf32>
//     return %3 : tensor<64x16xf32>
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
//     %1, %2, %3, %loop = transform.structured.tile_reduction_using_for %0
//       by tile_sizes = [0, 0, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
//     transform.apply_cse to %loop : !transform.any_op
//     transform.yield
//   }
// }

// // result:
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
//       // outer product
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
//       // inner product
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
//     transform.apply_cse to %forall : !transform.any_op
//     transform.apply_cse to %loop : !transform.any_op
//     transform.yield
//   }
// }

// // result:
// #map = affine_map<(d0) -> (d0 * 8)>
// #map1 = affine_map<(d0) -> (d0 * 32)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// #map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
// #map4 = affine_map<(d0) -> (d0 * 2)>
// module {
//   func.func @matmul_tile_parallel_reduction(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
//     %c8 = arith.constant 8 : index
//     %c4 = arith.constant 4 : index
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
//         %7 = affine.apply #map(%arg6)
//         %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %7] [8, 8] [1, 1] : tensor<8x32xf32> to tensor<8x8xf32>
//         %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[%7, 0] [8, 32] [1, 1] : tensor<32x32xf32> to tensor<8x32xf32>
//         %extracted_slice_5 = tensor.extract_slice %extracted_slice_2[0, 0] [8, 32] [1, 1] : tensor<8x32xf32> to tensor<8x32xf32>
//         %8 = linalg.matmul ins(%extracted_slice_3, %extracted_slice_4 : tensor<8x8xf32>, tensor<8x32xf32>) outs(%extracted_slice_5 : tensor<8x32xf32>) -> tensor<8x32xf32>
//         scf.forall.in_parallel {
//           tensor.parallel_insert_slice %8 into %arg7[0, 0, %arg6] [8, 32, 1] [1, 1, 1] : tensor<8x32xf32> into tensor<8x32x4xf32>
//         }
//       }
//       %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5 : tensor<8x32x4xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) {
//       ^bb0(%in: f32, %out: f32):
//         %7 = arith.addf %in, %out : f32
//         linalg.yield %7 : f32
//       } -> tensor<8x32xf32>
//       scf.forall.in_parallel {
//         tensor.parallel_insert_slice %6 into %arg5[%1, %2] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
//       }
//     }
//     return %0 : tensor<64x128xf32>
//   }
// }
