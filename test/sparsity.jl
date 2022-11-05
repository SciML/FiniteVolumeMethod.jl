function manual_construction(prob::FVMProblem)
    N = num_points(prob.mesh.points)
    mat = zeros(Bool, N, N)
    for i in 1:N
        mat[i, i] = true
        neigh_v = prob.mesh.neighbours.graph.N[i]
        for j in neigh_v
            if j ≠ DelaunayTriangulation.BoundaryIndex
                mat[i, j] = true
            end
        end
    end
    sparse(mat)
end

for prb in (DiffusionEquationOnASquarePlate, ReactionDiffusiondudt, DiffusionOnAWedge, TravellingWaveProblem)
    local prob
    prob, _ = prb()

    Jnew = FVM.jacobian_sparsity(prob)
    Jman = manual_construction(prob)

    @test Jnew == Jman

    if prb == TravellingWaveProblem
        @test BandedMatrix(Jman).l == 20
        @test BandedMatrix(Jnew).u == 20
    end
    @test sum(Jnew) == sum(Jman) == length(prob.mesh.adj.adjacent) + num_points(prob.mesh.points)
end