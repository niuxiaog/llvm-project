

func.func @tile_matmul_add(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<64x32xf32>, tensor<32x128xf32>) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = tensor.empty() : tensor<64x128xf32>
  %3 = linalg.add ins(%1, %arg2 : tensor<64x128xf32>, tensor<64x128xf32>) outs(%2 : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %3 : tensor<64x128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match ops{["linalg.add"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %tiled, %forall1 = transform.structured.tile_using_forall %1 tile_sizes [8, 32] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused, %forall2 = transform.structured.fuse_into_containing_op %0 into %forall1
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    %func_op = transform.structured.match ops{["func.func"]} in %arg1: (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func_op : !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op
    transform.yield
  }
}

#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @tile_matmul_add(%arg0: tensor<64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = tensor.empty() : tensor<64x128xf32>
    %1 = scf.forall (%arg3, %arg4) in (8, 4) shared_outs(%arg5 = %0) -> (tensor<64x128xf32>) {
      %2 = affine.apply #map(%arg3)
      %3 = affine.apply #map1(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%2, 0] [8, 32] [1, 1] : tensor<64x32xf32> to tensor<8x32xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %3] [32, 32] [1, 1] : tensor<32x128xf32> to tensor<32x32xf32>
      %extracted_slice_1 = tensor.extract_slice %0[%2, %3] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
      %4 = linalg.matmul ins(%extracted_slice, %extracted_slice_0 : tensor<8x32xf32>, tensor<32x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
      %extracted_slice_2 = tensor.extract_slice %arg2[%2, %3] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
      %extracted_slice_3 = tensor.extract_slice %arg5[%2, %3] [8, 32] [1, 1] : tensor<64x128xf32> to tensor<8x32xf32>
      %5 = linalg.add ins(%4, %extracted_slice_2 : tensor<8x32xf32>, tensor<8x32xf32>) outs(%extracted_slice_3 : tensor<8x32xf32>) -> tensor<8x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %5 into %arg5[%2, %3] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<64x128xf32>
      }
    }
    return %1 : tensor<64x128xf32>
  }
}
