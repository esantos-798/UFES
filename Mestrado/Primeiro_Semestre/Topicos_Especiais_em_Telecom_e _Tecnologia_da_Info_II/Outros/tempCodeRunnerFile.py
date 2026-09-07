def run_mohaf_ultra(problem_name, max_iter=500, num_pop=100, max_archive=100, p_turbulento=0.15):
    prob = get_problem(problem_name)
    D = prob.n_var
    lb, ub = prob.xl, prob.xu
    X = np.random.uniform(lb, ub, size=(num_pop, D))
    X_costs = np.array([prob.evaluate(ind) for ind in X])
    archive_pos, archive_costs = [], []
    for t in range(max_iter):
        for i in range(num_pop):
            sol_dominada = False
            eliminar_do_archive = []
            for j in range(len(archive_pos)):
                if domina(archive_costs[j], X_costs[i]): sol_dominada = True; break
                elif domina(X_costs[i], archive_costs[j]): eliminar_do_archive.append(j)
            if not sol_dominada:
                for idx in sorted(eliminar_do_archive, reverse=True):
                    archive_pos.pop(idx); archive_costs.pop(idx)
                archive_pos.append(np.copy(X[i])); archive_costs.append(np.copy(X_costs[i]))
        if len(archive_pos) > max_archive:
            dist = calcular_crowding_distance(np.array(archive_costs))
            idx_remover = np.argsort(dist)[:(len(archive_pos) - max_archive)]
            for idx in sorted(idx_remover, reverse=True): archive_pos.pop(idx); archive_costs.pop(idx)
        a = 2.0 * np.log(1.0 + (max_iter - t) / max_iter) / np.log(2.0)
        for i in range(num_pop):
            if np.random.rand() < p_turbulento:
                escala_vortice = 0.02 * (ub - lb) * (1.0 - (t / max_iter))
                X[i] = X[i] + np.random.standard_cauchy(D) * escala_vortice
            else:
                if len(archive_pos) > 0: best_pos = selecionar_lider_roleta(archive_pos, np.array(archive_costs))
                else: best_pos = X[np.random.randint(0, num_pop)]
                r1, r2 = np.random.rand(D), np.random.rand(D)
                A, C = 2.0 * a * r1 - a, 2.0 * r2 * np.cos(np.pi * (t / max_iter))
                X_vizinho = X[np.random.randint(0, num_pop)]
                X[i] = X_vizinho - A * np.abs(C * X_vizinho - X[i]) if np.abs(A[0]) > 1.0 else best_pos - A * np.abs(C * best_pos - X[i])
            X[i] = np.clip(X[i], lb, ub)
            X_costs[i] = prob.evaluate(X[i])
    return np.array(archive_costs)