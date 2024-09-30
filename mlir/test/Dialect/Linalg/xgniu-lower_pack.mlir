// RUN: ./llvm-install/bin/mlir-opt mlir/test/Dialect/Linalg/lower_pack.mlir -transform-interpreter -split-input-file -debug

func.func @entry(%arg1: tensor<512x256xbf16>) -> tensor<512x256xbf16> {
    %1 = tensor.empty() : tensor<8x16x32x32xbf16>
    %packed = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xbf16> -> tensor<8x16x32x32xbf16>
    %2 = tensor.empty() : tensor<8x16x16x32x2xbf16>
    %packed_packed = tensor.pack %packed inner_dims_pos = [2] inner_tiles = [2] into %2 : tensor<8x16x32x32xbf16> -> tensor<8x16x16x32x2xbf16>

    %3 = tensor.empty() : tensor<8x16x32x32xbf16>
    %unpacked = tensor.unpack %packed_packed inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<8x16x16x32x2xbf16> -> tensor<8x16x32x32xbf16>
    %4 = tensor.empty() : tensor<512x256xbf16>
    %unpacked_unpacked = tensor.unpack %unpacked outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %4 : tensor<8x16x32x32xbf16> -> tensor<512x256xbf16>
    return %unpacked_unpacked : tensor<512x256xbf16>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %pack = transform.structured.match ops{["tensor.pack"]} in %arg1
            : (!transform.any_op) -> !transform.op<"tensor.pack">
        transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
            -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

        %unpack = transform.structured.match ops{["tensor.unpack"]} in %arg1
            : (!transform.any_op) -> !transform.op<"tensor.unpack">
        transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
            -> (!transform.op<"tensor.empty">,
                !transform.op<"linalg.transpose">,
                !transform.op<"tensor.collapse_shape">,
                !transform.op<"tensor.extract_slice">)

        %func_op = transform.structured.match ops{["func.func"]} in %arg1
            : (!transform.any_op) -> !transform.any_op
        transform.apply_cse to %func_op : !transform.any_op
        transform.apply_dce to %func_op : !transform.any_op
        transform.yield
    }
}

// result:
// module {
//   func.func @entry(%arg0: tensor<512x256xbf16>) -> tensor<512x256xbf16> {
//     %0 = tensor.empty() : tensor<8x16x32x32xbf16>
//     %cst = arith.constant 0.000000e+00 : bf16
//     %padded = tensor.pad %arg0 low[0, 0] high[0, 0] {
//     ^bb0(%arg1: index, %arg2: index):
//       tensor.yield %cst : bf16
//     } : tensor<512x256xbf16> to tensor<512x256xbf16>
//     %expanded = tensor.expand_shape %padded [[0, 1], [2, 3]] output_shape [16, 32, 8, 32] : tensor<512x256xbf16> into tensor<16x32x8x32xbf16>
//     %transposed = linalg.transpose ins(%expanded : tensor<16x32x8x32xbf16>) outs(%0 : tensor<8x16x32x32xbf16>) permutation = [2, 0, 1, 3]

//     %1 = tensor.empty() : tensor<8x16x16x32x2xbf16>
//     %padded_0 = tensor.pad %transposed low[0, 0, 0, 0] high[0, 0, 0, 0] {
//     ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
//       tensor.yield %cst : bf16
//     } : tensor<8x16x32x32xbf16> to tensor<8x16x32x32xbf16>
//     %expanded_1 = tensor.expand_shape %padded_0 [[0], [1], [2, 3], [4]] output_shape [8, 16, 16, 2, 32] : tensor<8x16x32x32xbf16> into tensor<8x16x16x2x32xbf16>
//     %transposed_2 = linalg.transpose ins(%expanded_1 : tensor<8x16x16x2x32xbf16>) outs(%1 : tensor<8x16x16x32x2xbf16>) permutation = [0, 1, 2, 4, 3]

//     %2 = tensor.empty() : tensor<8x16x16x2x32xbf16>
//     %transposed_3 = linalg.transpose ins(%transposed_2 : tensor<8x16x16x32x2xbf16>) outs(%2 : tensor<8x16x16x2x32xbf16>) permutation = [0, 1, 2, 4, 3]
//     %collapsed = tensor.collapse_shape %transposed_3 [[0], [1], [2, 3], [4]] : tensor<8x16x16x2x32xbf16> into tensor<8x16x32x32xbf16>
//     %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 0] [8, 16, 32, 32] [1, 1, 1, 1] : tensor<8x16x32x32xbf16> to tensor<8x16x32x32xbf16>
//     %3 = linalg.copy ins(%extracted_slice : tensor<8x16x32x32xbf16>) outs(%0 : tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>

//     %4 = tensor.empty() : tensor<512x256xbf16>
//     %5 = tensor.empty() : tensor<16x32x8x32xbf16>
//     %transposed_4 = linalg.transpose ins(%3 : tensor<8x16x32x32xbf16>) outs(%5 : tensor<16x32x8x32xbf16>) permutation = [1, 2, 0, 3]
//     %collapsed_5 = tensor.collapse_shape %transposed_4 [[0, 1], [2, 3]] : tensor<16x32x8x32xbf16> into tensor<512x256xbf16>
//     %extracted_slice_6 = tensor.extract_slice %collapsed_5[0, 0] [512, 256] [1, 1] : tensor<512x256xbf16> to tensor<512x256xbf16>
//     %6 = linalg.copy ins(%extracted_slice_6 : tensor<512x256xbf16>) outs(%4 : tensor<512x256xbf16>) -> tensor<512x256xbf16>
//     return %6 : tensor<512x256xbf16>
//   }
// }






// #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
// #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
// #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// #map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// #map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// func.func @entry(%arg0: tensor<64x512xbf16>, %arg1: tensor<512x256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<256x1024xbf16>, %arg4: tensor<1024xbf16>) -> tensor<64x1024xbf16> attributes {llvm.emit_c_interface, runtime_const_args_index = [1 : i32, 2 : i32, 3 : i32, 4 : i32]} {
//     %1 = tensor.empty() : tensor<2x16x32x32xbf16>
//     %packed_arg0 = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<64x512xbf16> -> tensor<2x16x32x32xbf16>
//     %2 = tensor.empty() : tensor<8x16x32x32xbf16>
//     %packed_arg1 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<512x256xbf16> -> tensor<8x16x32x32xbf16>
//     %3 = tensor.empty() : tensor<8x16x16x32x2xbf16>
//     %packed_packed_arg1 = tensor.pack %packed_arg1 inner_dims_pos = [2] inner_tiles = [2] into %3 : tensor<8x16x32x32xbf16> -> tensor<8x16x16x32x2xbf16>
//     %4 = tensor.empty() : tensor<2x8x32x32xbf16>
//     %cst_0 = arith.constant 0.000000e+00 : bf16
//     %5 = linalg.fill ins(%cst_0 : bf16) outs(%4 : tensor<2x8x32x32xbf16>) -> tensor<2x8x32x32xbf16>
//     %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%packed_arg0, %packed_packed_arg1 : tensor<2x16x32x32xbf16>, tensor<8x16x16x32x2xbf16>) outs(%5 : tensor<2x8x32x32xbf16>) {
//     ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
//       %44 = arith.mulf %in, %in_0 : bf16
//       %55 = arith.addf %out, %44 : bf16
//       linalg.yield %55 : bf16
//     } -> tensor<2x8x32x32xbf16>
//     %15 = tensor.empty() : tensor<8x32xbf16>
//     %packed_arg2 = tensor.pack %arg2 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %15 : tensor<256xbf16> -> tensor<8x32xbf16>
//     %bc_arg2_init = tensor.empty() : tensor<2x8x32x32xbf16>
//     %bc_arg2 = linalg.broadcast ins(%packed_arg2 : tensor<8x32xbf16>) outs(%bc_arg2_init : tensor<2x8x32x32xbf16>) dimensions = [0, 2]
//     %7 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bc_arg2 : tensor<2x8x32x32xbf16>) outs(%6 : tensor<2x8x32x32xbf16>) {
//     ^bb0(%in: bf16, %out: bf16):
//       %45 = arith.addf %in, %out : bf16
//       linalg.yield %45 : bf16
//     } -> tensor<2x8x32x32xbf16>
//     %8 = tensor.empty() : tensor<32x8x32x32xbf16>
//     %packed_arg3 = tensor.pack %arg3 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %8 : tensor<256x1024xbf16> -> tensor<32x8x32x32xbf16>
//     %9 = tensor.empty() : tensor<32x8x16x32x2xbf16>
//     %packed_packed_arg3 = tensor.pack %packed_arg3 inner_dims_pos = [2] inner_tiles = [2] into %9 : tensor<32x8x32x32xbf16> -> tensor<32x8x16x32x2xbf16>
//     %10 = tensor.empty() : tensor<2x32x32x32xbf16>
//     %11 = linalg.fill ins(%cst_0 : bf16) outs(%10 : tensor<2x32x32x32xbf16>) -> tensor<2x32x32x32xbf16>
//     %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%7, %packed_packed_arg3 : tensor<2x8x32x32xbf16>, tensor<32x8x16x32x2xbf16>) outs(%11 : tensor<2x32x32x32xbf16>) {
//     ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
//       %46 = arith.mulf %in, %in_0 : bf16
//       %56 = arith.addf %out, %46 : bf16
//       linalg.yield %56 : bf16
//     } -> tensor<2x32x32x32xbf16>
//     %16 = tensor.empty() : tensor<32x32xbf16>
//     %packed_arg4 = tensor.pack %arg4 outer_dims_perm = [0] inner_dims_pos = [0] inner_tiles = [32] into %16 : tensor<1024xbf16> -> tensor<32x32xbf16>
//     %13 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%packed_arg4 : tensor<32x32xbf16>) outs(%12 : tensor<2x32x32x32xbf16>) {
//     ^bb0(%in: bf16, %out: bf16):
//       %47 = arith.addf %in, %out : bf16
//       linalg.yield %47 : bf16
//     } -> tensor<2x32x32x32xbf16>
//     %14 = tensor.empty() : tensor<64x1024xbf16>
//     %unpack = tensor.unpack %13 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %14 : tensor<2x32x32x32xbf16> -> tensor<64x1024xbf16>
//     return %unpack : tensor<64x1024xbf16>
// }

// module attributes {transform.with_named_sequence} {
//     transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
//         %pack = transform.structured.match ops{["tensor.pack"]} in %arg1
//             : (!transform.any_op) -> !transform.op<"tensor.pack">
//         transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
//             -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

//         %unpack = transform.structured.match ops{["tensor.unpack"]} in %arg1
//             : (!transform.any_op) -> !transform.op<"tensor.unpack">
//         transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
//             -> (!transform.op<"tensor.empty">,
//                 !transform.op<"linalg.transpose">,
//                 !transform.op<"tensor.collapse_shape">,
//                 !transform.op<"tensor.extract_slice">)

//         %func_op = transform.structured.match ops{["func.func"]} in %arg1
//             : (!transform.any_op) -> !transform.any_op
//         transform.apply_cse to %func_op : !transform.any_op
//         transform.apply_dce to %func_op : !transform.any_op
//         transform.yield
//     }
// }
