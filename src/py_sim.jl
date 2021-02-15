module py_sim

using PyCall
np = pyimport("numpy")

function T(mu, N)
	Im = np.zeros((N, N))
    np.fill_diagonal(Im, 1)
    Im = Im .* im
    v = np.ones(N-1)

    # Tight binding Hamiltonian
    H = np.diag(-v, -1) + np.diag(4*np.ones(N)-mu, 0) + np.diag(-v, 1)
    T11 = -Im + 0.5 * H
    T12 = -0.5 * H
    T21 = -0.5 * H
    T22 = Im + 0.5 * H

    return np.array(np.bmat([[T11, T12], [T21, T22]]))
end

out = T(0.5, 40)

end
