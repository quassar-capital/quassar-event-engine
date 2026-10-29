
# Research 02 — Independent Julia numerical reference kernels.
#
# Deterministic abnormal returns, cumulative abnormal returns,
# and Holm multiple-testing adjustment.
#
# Julia standard library only.
# No causal interpretation or finite-sample inferential
# guarantee is implied by these numerical operations.

using Test

module QuassarResearch02

export abnormal_returns, cumulative_abnormal_return, holm_adjustment

function abnormal_returns(
    observed::AbstractVector{<:Real},
    expected::AbstractVector{<:Real},
)
    isempty(observed) &&
        throw(ArgumentError("Inputs must be nonempty"))

    length(observed) == length(expected) ||
        throw(DimensionMismatch("Input dimensions must match"))

    all(isfinite, observed) && all(isfinite, expected) ||
        throw(ArgumentError("Inputs must be finite"))

    actual = Float64.(observed)
    baseline = Float64.(expected)

    result = actual .- baseline

    all(isfinite, result) ||
        throw(OverflowError("Abnormal-return overflow"))

    return result
end

function cumulative_abnormal_return(
    abnormal::AbstractVector{<:Real},
    start::Integer,
    stop::Integer,
)
    # Zero-based, half-open event window [start, stop),
    # matching the Rust and C++ reference implementations.

    0 <= start < stop <= length(abnormal) ||
        throw(ArgumentError("Invalid half-open event window"))

    all(isfinite, abnormal) ||
        throw(ArgumentError("Abnormal returns must be finite"))

    values = Float64.(abnormal)

    result = sum(@view values[(start + 1):stop])

    isfinite(result) ||
        throw(OverflowError("CAR calculation overflow"))

    return result
end

function holm_adjustment(
    pvalues::AbstractVector{<:Real},
)
    isempty(pvalues) &&
        throw(ArgumentError("At least one p-value is required"))

    all(p -> isfinite(p) && 0 <= p <= 1, pvalues) ||
        throw(ArgumentError("P-values must lie in [0, 1]"))

    values = Float64.(pvalues)
    m = length(values)

    order = sortperm(collect(1:m); by=i -> (values[i], i))

    adjusted = zeros(Float64, m)
    running_max = 0.0

    for rank in 1:m
        index = order[rank]
        multiplier = m - rank + 1

        candidate = min(1.0, values[index] * multiplier)
        running_max = max(running_max, candidate)

        adjusted[index] = running_max
    end

    return adjusted
end

end # module QuassarResearch02


# Independent deterministic reference tests.
# Run with: julia research_02/julia/event_inference.jl

if abspath(PROGRAM_FILE) == @__FILE__

    using Test
    using .QuassarResearch02

    @testset "Research 02 — Julia numerical kernels" begin

        @testset "Abnormal returns" begin
            actual = [0.03, -0.01, 0.04]
            expected = [0.01, 0.00, 0.02]

            result = abnormal_returns(actual, expected)

            @test all(
                isapprox.(result, [0.02, -0.01, 0.02]; atol=1e-12)
            )

            @test_throws DimensionMismatch abnormal_returns(
                [0.01, 0.02], [0.01]
            )
        end

        @testset "Cumulative abnormal returns" begin
            abnormal = [0.01, 0.02, -0.01, 0.03]

            result = cumulative_abnormal_return(
                abnormal, 1, 4
            )

            @test isapprox(result, 0.04; atol=1e-12)

            @test_throws ArgumentError cumulative_abnormal_return(
                abnormal, 3, 2
            )
        end

        @testset "Holm multiple-testing adjustment" begin
            pvalues = [0.01, 0.04, 0.03, 0.20]

            result = holm_adjustment(pvalues)

            expected = [0.04, 0.09, 0.09, 0.20]

            @test all(
                isapprox.(result, expected; atol=1e-12)
            )

            @test_throws ArgumentError holm_adjustment(
                [0.01, 1.2]
            )
        end

    end
end
