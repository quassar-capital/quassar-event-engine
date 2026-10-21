module QuassarEventEngine

export logsumexp

"""
    logsumexp(values)

Compute the log of the sum of exponentials without naive overflow.
"""
function logsumexp(values::AbstractVector{<:Real})
    isempty(values) && throw(ArgumentError("Input must be nonempty"))
    any(isnan, values) && throw(ArgumentError("NaN is not permitted"))
    m = maximum(values)
    m == Inf && return Inf
    m == -Inf && return -Inf
    return m + log(sum(exp(x - m) for x in values))
end

end
