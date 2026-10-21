using Test
using QuassarEventEngine

@testset "Stable logsumexp" begin
    @test logsumexp([0.0, 0.0]) ≈ log(2)
    @test isfinite(logsumexp([1000.0, 1001.0]))
    @test logsumexp([-Inf, -Inf]) == -Inf
    @test_throws ArgumentError logsumexp(Float64[])
end
