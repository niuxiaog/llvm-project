#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module attributes {torch.debug_module_name = "ConstantWeightLinear"} {
  func.func @forward(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x4xf32> attributes {linalg.const_args = [1 : i32]} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %cst_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4x8xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.mulf %in, %cst_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4x8xf32>
    %3 = tensor.empty() : tensor<8x4xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4x8xf32>) outs(%3 : tensor<8x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x4xf32>
    %5 = tensor.empty() : tensor<4x4xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %7 = linalg.matmul ins(%1, %4 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%6 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %7 : tensor<4x4xf32>
  }
}