"""
Técnicas Avanzadas de Tratamiento de Señal en Comunicaciones 2025-2026
Laboratorio #1: Filtrado de partículas

Implementación completa:
  - Simulador de trayectoria (sistema lineal dinámico)
  - Filtro de Kalman
  - Filtro de partículas (SIS con remuestreo)
  - Experimentos comparativos y gráficos

Supuestos razonables adoptados:
  - Número de partículas por defecto: N = 500
  - Número de simulaciones para promediar MSE: 50
  - Horizonte temporal: T = 100
  - Ruido uniforme (apartado 3): U(-a, a) con a = σ·√3 para mantener la misma varianza
  - Remuestreo sistemático (eficiente y de baja varianza)
  - Distribución de importancia: prior p(x_t | x_{t-1})
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import inv, det

# ===========================================================================
# Parámetros del sistema
# ===========================================================================
A = np.array([[0.9, 0.12],
              [0.08, 0.85]])

B = np.array([[-0.12, 0.15],
              [0.4, -0.15]])

T = 100  # Horizonte temporal

# ===========================================================================
# 1. Simulador del sistema dinámico lineal
# ===========================================================================

def simulate_system(A, B, sigma_u, sigma_v, T, noise_type='gaussian'):
    """
    Genera la trayectoria real del barco y las observaciones del sónar.

    Args:
        A: Matriz de transición (2x2).
        B: Matriz de observación (2x2).
        sigma_u: Desviación estándar del ruido de sistema.
        sigma_v: Desviación estándar del ruido de observación.
        T: Número de instantes temporales.
        noise_type: 'gaussian' o 'uniform'.

    Returns:
        x: Estados reales, shape (2, T+1), con x[:, 0] = x0.
        y: Observaciones, shape (2, T).
    """
    d_x = A.shape[0]
    d_y = B.shape[0]

    x = np.zeros((d_x, T + 1))
    y = np.zeros((d_y, T))

    # Estado inicial: x0 ~ N(0, 25·I2)
    x[:, 0] = np.random.multivariate_normal(np.zeros(d_x), 25.0 * np.eye(d_x))

    for t in range(1, T + 1):
        # Ruido de sistema
        if noise_type == 'gaussian':
            u_t = np.random.normal(0, sigma_u, size=d_x)
        else:  # uniform con la misma varianza: U(-a, a), a = sigma * sqrt(3)
            a_u = sigma_u * np.sqrt(3)
            u_t = np.random.uniform(-a_u, a_u, size=d_x)

        x[:, t] = A @ x[:, t - 1] + u_t

        # Ruido de observación
        if noise_type == 'gaussian':
            v_t = np.random.normal(0, sigma_v, size=d_y)
        else:
            a_v = sigma_v * np.sqrt(3)
            v_t = np.random.uniform(-a_v, a_v, size=d_y)

        y[:, t - 1] = B @ x[:, t] + v_t

    return x, y


# ===========================================================================
# 2. Filtro de Kalman
# ===========================================================================

def kalman_filter(y, A, B, sigma_u, sigma_v):
    """
    Filtro de Kalman para el sistema lineal.

    Args:
        y: Observaciones, shape (2, T).
        A, B: Matrices del sistema.
        sigma_u, sigma_v: Desviaciones estándar de los ruidos.

    Returns:
        x_est: Estimaciones filtradas, shape (2, T+1).
        P_hist: Covarianzas filtradas, lista de matrices (T+1).
    """
    d_x = A.shape[0]
    T = y.shape[1]

    Q = (sigma_u ** 2) * np.eye(d_x)   # Covarianza ruido de sistema
    R = (sigma_v ** 2) * np.eye(d_x)   # Covarianza ruido de observación

    # Inicialización con la prior
    x_filt = np.zeros(d_x)
    P_filt = 25.0 * np.eye(d_x)

    x_est = np.zeros((d_x, T + 1))
    P_hist = [None] * (T + 1)
    x_est[:, 0] = x_filt
    P_hist[0] = P_filt.copy()

    for t in range(1, T + 1):
        # --- Predicción ---
        x_pred = A @ x_filt
        P_pred = A @ P_filt @ A.T + Q

        # --- Actualización ---
        S = B @ P_pred @ B.T + R                      # Covarianza de innovación
        K = P_pred @ B.T @ inv(S)                     # Ganancia de Kalman
        innov = y[:, t - 1] - B @ x_pred              # Innovación
        x_filt = x_pred + K @ innov
        P_filt = (np.eye(d_x) - K @ B) @ P_pred

        x_est[:, t] = x_filt
        P_hist[t] = P_filt.copy()

    return x_est, P_hist


# ===========================================================================
# 3. Filtro de partículas (SIS con remuestreo)
# ===========================================================================

def systematic_resampling(weights):
    """
    Remuestreo sistemático (baja varianza).

    Args:
        weights: Vector de pesos normalizados, shape (N,).

    Returns:
        indices: Índices de las partículas seleccionadas.
    """
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return indices


def particle_filter(y, A, B, sigma_u, sigma_v, N_particles):
    """
    Filtro de partículas con SIS y remuestreo sistemático.
    Distribución de importancia: prior p(x_t | x_{t-1}).

    Args:
        y: Observaciones, shape (2, T).
        A, B: Matrices del sistema.
        sigma_u, sigma_v: Desviaciones estándar de los ruidos.
        N_particles: Número de partículas.

    Returns:
        x_est: Estimaciones (media ponderada), shape (2, T+1).
        particles_final: Partículas en t=T, shape (2, N_particles).
        weights_final: Pesos en t=T, shape (N_particles,).
    """
    d_x = A.shape[0]
    T = y.shape[1]

    R = (sigma_v ** 2) * np.eye(d_x)
    R_inv = inv(R)
    R_det = det(R)
    log_norm_const = -0.5 * d_x * np.log(2 * np.pi) - 0.5 * np.log(R_det)

    # Inicialización: muestrear de la prior x0 ~ N(0, 25·I2)
    particles = np.random.multivariate_normal(
        np.zeros(d_x), 25.0 * np.eye(d_x), size=N_particles
    ).T  # shape (2, N_particles)

    weights = np.ones(N_particles) / N_particles

    x_est = np.zeros((d_x, T + 1))
    x_est[:, 0] = np.mean(particles, axis=1)

    for t in range(1, T + 1):
        # --- Propagación con la prior: x_t^(i) ~ N(A·x_{t-1}^(i), σ_u²·I) ---
        noise = np.random.normal(0, sigma_u, size=(d_x, N_particles))
        particles = A @ particles + noise

        # --- Actualización de pesos: w ∝ p(y_t | x_t^(i)) ---
        residuals = y[:, t - 1:t] - B @ particles  # shape (2, N_particles)

        # Log-verosimilitud Gaussiana
        log_w = log_norm_const - 0.5 * np.sum(residuals * (R_inv @ residuals), axis=0)
        log_w -= np.max(log_w)  # Estabilidad numérica
        weights = np.exp(log_w)
        weights /= np.sum(weights)

        # Estimación como media ponderada
        x_est[:, t] = particles @ weights

        # --- Remuestreo ---
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < N_particles / 2:
            idx = systematic_resampling(weights)
            particles = particles[:, idx]
            weights = np.ones(N_particles) / N_particles

    return x_est, particles, weights


def particle_filter_uniform(y, A, B, sigma_u, sigma_v, N_particles):
    """
    Filtro de partículas adaptado a ruido uniforme.
    Verosimilitud: indicadora (producto de distribuciones uniformes).

    Args:
        y, A, B, sigma_u, sigma_v, N_particles: Igual que particle_filter.

    Returns:
        x_est: Estimaciones, shape (2, T+1).
    """
    d_x = A.shape[0]
    T = y.shape[1]

    a_u = sigma_u * np.sqrt(3)
    a_v = sigma_v * np.sqrt(3)

    # Inicialización
    particles = np.random.multivariate_normal(
        np.zeros(d_x), 25.0 * np.eye(d_x), size=N_particles
    ).T

    weights = np.ones(N_particles) / N_particles
    x_est = np.zeros((d_x, T + 1))
    x_est[:, 0] = np.mean(particles, axis=1)

    for t in range(1, T + 1):
        # Propagación con ruido uniforme
        noise = np.random.uniform(-a_u, a_u, size=(d_x, N_particles))
        particles = A @ particles + noise

        # Verosimilitud uniforme: p(y|x) ∝ 1 si |y - Bx| <= a_v en cada componente
        residuals = y[:, t - 1:t] - B @ particles
        inside = np.all(np.abs(residuals) <= a_v, axis=0).astype(float)

        if np.sum(inside) == 0:
            # Ninguna partícula cae dentro → usar inversa de distancia como respaldo
            dist = np.sum(np.abs(residuals), axis=0)
            weights = 1.0 / (dist + 1e-10)
        else:
            weights = inside

        weights /= np.sum(weights)
        x_est[:, t] = particles @ weights

        # Remuestreo
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < N_particles / 2:
            idx = systematic_resampling(weights)
            particles = particles[:, idx]
            weights = np.ones(N_particles) / N_particles

    return x_est, particles, weights


# ===========================================================================
# 4. Estimación directa invirtiendo B (para comparación)
# ===========================================================================

def direct_estimation(y, B):
    """
    Estima x directamente como B^{-1} y (sin filtrado).

    Returns:
        x_est: shape (2, T). Solo para t=1..T (no hay estimación para t=0).
    """
    B_inv = inv(B)
    return B_inv @ y


# ===========================================================================
# 5. Estimación de Var(z_100 | y_{1:100}) con filtro de partículas
# ===========================================================================

def estimate_variance_z100(y, A, B, sigma_u, sigma_v, N_particles):
    """
    Estima la varianza de z_100 = ||x_100|| = sqrt(x_100^T x_100)
    condicionada a y_{1:100}, usando las partículas del filtro.

    Returns:
        var_z100: Varianza estimada.
        mean_z100: Media estimada.
        z_particles: Valores de z para cada partícula.
        weights: Pesos finales.
    """
    _, particles, weights = particle_filter(y, A, B, sigma_u, sigma_v, N_particles)

    # z_100^(i) = ||x_100^(i)||
    z_particles = np.sqrt(np.sum(particles ** 2, axis=0))

    # Media y varianza ponderadas
    mean_z = np.sum(weights * z_particles)
    var_z = np.sum(weights * (z_particles - mean_z) ** 2)

    return var_z, mean_z, z_particles, weights


# ===========================================================================
# EXPERIMENTOS Y GRÁFICOS
# ===========================================================================

def run_all_experiments():
    """Ejecuta todos los experimentos requeridos y genera los gráficos."""

    np.random.seed(42)
    output_dir = ""  # Directorio actual

    # Configuración de figuras
    plt.rcParams.update({
        'figure.figsize': (12, 5),
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # ===================================================================
    # APARTADO 1: Valores reales vs estimados (σ_u=1, σ_v=1, N=500)
    # ===================================================================
    print("=" * 60)
    print("APARTADO 1: Trayectoria real vs estimaciones")
    print("=" * 60)

    sigma_u, sigma_v = 1.0, 1.0
    N_particles = 500

    x_true, y_obs = simulate_system(A, B, sigma_u, sigma_v, T)
    x_kalman, _ = kalman_filter(y_obs, A, B, sigma_u, sigma_v)
    x_pf, _, _ = particle_filter(y_obs, A, B, sigma_u, sigma_v, N_particles)

    time_axis = np.arange(T + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, ax in enumerate(axes):
        ax.plot(time_axis, x_true[i, :], 'k-', linewidth=1.5, label='Real')
        ax.plot(time_axis, x_kalman[i, :], 'b--', linewidth=1.2, label='Kalman')
        ax.plot(time_axis, x_pf[i, :], 'r:', linewidth=1.2, label=f'Partículas (N={N_particles})')
        ax.set_xlabel('Tiempo t')
        ax.set_ylabel(f'$x_{{{i+1},t}}$')
        ax.set_title(f'Componente $x_{{{i+1}}}$ — $\\sigma_u={sigma_u}$, $\\sigma_v={sigma_v}$')
        ax.legend()
    plt.tight_layout()
    plt.savefig('fig1_trayectoria_estimada.png', dpi=150)
    plt.close()
    print("  → fig1_trayectoria_estimada.png generada.")

    # ===================================================================
    # APARTADO 2: Efecto de σ_u y σ_v
    # ===================================================================
    print("\n" + "=" * 60)
    print("APARTADO 2: Comparación σ_u=0.5/σ_v=5 vs σ_u=5/σ_v=0.5")
    print("=" * 60)

    configs = [
        (0.5, 5.0, 'Caso 1: $\\sigma_u=0.5$, $\\sigma_v=5$'),
        (5.0, 0.5, 'Caso 2: $\\sigma_u=5$, $\\sigma_v=0.5$'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, (su, sv, title) in enumerate(configs):
        x_true_c, y_obs_c = simulate_system(A, B, su, sv, T)
        x_kalman_c, _ = kalman_filter(y_obs_c, A, B, su, sv)
        x_pf_c, _, _ = particle_filter(y_obs_c, A, B, su, sv, N_particles)

        for col in range(2):
            ax = axes[row, col]
            ax.plot(time_axis, x_true_c[col, :], 'k-', linewidth=1.5, label='Real')
            ax.plot(time_axis, x_kalman_c[col, :], 'b--', linewidth=1.2, label='Kalman')
            ax.plot(time_axis, x_pf_c[col, :], 'r:', linewidth=1.2, label=f'Partículas')
            ax.set_xlabel('Tiempo t')
            ax.set_ylabel(f'$x_{{{col+1},t}}$')
            ax.set_title(f'{title} — Componente $x_{{{col+1}}}$')
            ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('fig2_efecto_sigma.png', dpi=150)
    plt.close()
    print("  → fig2_efecto_sigma.png generada.")

    # ===================================================================
    # APARTADO 3: MSE promediado sobre múltiples simulaciones
    # ===================================================================
    print("\n" + "=" * 60)
    print("APARTADO 3: Error cuadrático medio (MSE)")
    print("=" * 60)

    sigma_u, sigma_v = 1.0, 1.0
    N_sim = 50
    N_particles_list = [100, 500]

    mse_kalman_acc = np.zeros(T + 1)
    mse_pf_acc = {N: np.zeros(T + 1) for N in N_particles_list}
    mse_direct_acc = np.zeros(T)

    print(f"  Promediando sobre {N_sim} simulaciones...")
    for sim in range(N_sim):
        if (sim + 1) % 10 == 0:
            print(f"    Simulación {sim + 1}/{N_sim}")

        x_true_s, y_obs_s = simulate_system(A, B, sigma_u, sigma_v, T)

        # Kalman
        x_k, _ = kalman_filter(y_obs_s, A, B, sigma_u, sigma_v)
        mse_kalman_acc += np.sum((x_true_s - x_k) ** 2, axis=0)

        # Partículas
        for N_p in N_particles_list:
            x_p, _, _ = particle_filter(y_obs_s, A, B, sigma_u, sigma_v, N_p)
            mse_pf_acc[N_p] += np.sum((x_true_s - x_p) ** 2, axis=0)

        # Estimación directa (B^{-1} y)
        x_direct = direct_estimation(y_obs_s, B)
        mse_direct_acc += np.sum((x_true_s[:, 1:] - x_direct) ** 2, axis=0)

    mse_kalman = mse_kalman_acc / N_sim
    mse_pf = {N: mse_pf_acc[N] / N_sim for N in N_particles_list}
    mse_direct = mse_direct_acc / N_sim

    # Gráfico MSE
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_axis, mse_kalman, 'b-', linewidth=1.5, label='Kalman')
    colors_pf = ['r', 'orange']
    for idx, N_p in enumerate(N_particles_list):
        ax.plot(time_axis, mse_pf[N_p], linestyle='--', color=colors_pf[idx],
                linewidth=1.2, label=f'Partículas (N={N_p})')
    ax.plot(np.arange(1, T + 1), mse_direct, 'g:', linewidth=1.2, label='Directa ($B^{-1}y$)')
    ax.set_xlabel('Tiempo t')
    ax.set_ylabel('MSE')
    ax.set_title(f'Error cuadrático medio — $\\sigma_u={sigma_u}$, $\\sigma_v={sigma_v}$ (promedio de {N_sim} sim.)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig3_mse_comparacion.png', dpi=150)
    plt.close()
    print("  → fig3_mse_comparacion.png generada.")

    # MSE medio global
    print(f"\n  MSE medio global (t=1..{T}):")
    print(f"    Kalman:              {np.mean(mse_kalman[1:]):.4f}")
    for N_p in N_particles_list:
        print(f"    Partículas (N={N_p}):  {np.mean(mse_pf[N_p][1:]):.4f}")
    print(f"    Directa (B⁻¹y):    {np.mean(mse_direct):.4f}")

    # ===================================================================
    # APARTADO 3b: Ruido uniforme
    # ===================================================================
    print("\n" + "-" * 60)
    print("APARTADO 3b: Ruido uniforme (cuantificación)")
    print("-" * 60)

    N_sim_unif = 50
    N_p_unif = 500

    mse_kalman_unif_acc = np.zeros(T + 1)
    mse_pf_gauss_on_unif_acc = np.zeros(T + 1)
    mse_pf_unif_acc = np.zeros(T + 1)

    for sim in range(N_sim_unif):
        x_true_u, y_obs_u = simulate_system(A, B, sigma_u, sigma_v, T, noise_type='uniform')

        # Kalman (asume Gaussiano, modelo mal especificado)
        x_k_u, _ = kalman_filter(y_obs_u, A, B, sigma_u, sigma_v)
        mse_kalman_unif_acc += np.sum((x_true_u - x_k_u) ** 2, axis=0)

        # Filtro de partículas asumiendo Gaussiano (modelo mal especificado)
        x_pf_g, _, _ = particle_filter(y_obs_u, A, B, sigma_u, sigma_v, N_p_unif)
        mse_pf_gauss_on_unif_acc += np.sum((x_true_u - x_pf_g) ** 2, axis=0)

        # Filtro de partículas adaptado a uniforme
        x_pf_u, _, _ = particle_filter_uniform(y_obs_u, A, B, sigma_u, sigma_v, N_p_unif)
        mse_pf_unif_acc += np.sum((x_true_u - x_pf_u) ** 2, axis=0)

    mse_kalman_unif = mse_kalman_unif_acc / N_sim_unif
    mse_pf_gauss_on_unif = mse_pf_gauss_on_unif_acc / N_sim_unif
    mse_pf_unif = mse_pf_unif_acc / N_sim_unif

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_axis, mse_kalman_unif, 'b-', linewidth=1.5, label='Kalman')
    ax.plot(time_axis, mse_pf_gauss_on_unif, 'r--', linewidth=1.2, label='Partículas gaussiano')
    ax.plot(time_axis, mse_pf_unif, 'm:', linewidth=1.5, label='Partículas uniforme')
    ax.set_xlabel('Tiempo t')
    ax.set_ylabel('MSE')
    ax.set_title(f'MSE con ruido uniforme — $\\sigma_u={sigma_u}$, $\\sigma_v={sigma_v}$ (promedio de {N_sim_unif} sim.)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig4_mse_uniforme.png', dpi=150)
    plt.close()
    print("  → fig4_mse_uniforme.png generada.")

    # MSE medio global
    print(f"\n  MSE medio global (t=1..{T}):")
    print(f"    Kalman:                       {np.mean(mse_kalman_unif[1:]):.4f}")
    print(f"    Partículas gaussiano (N={N_p_unif}): {np.mean(mse_pf_gauss_on_unif[1:]):.4f}")
    print(f"    Partículas uniforme (N={N_p_unif}):  {np.mean(mse_pf_unif[1:]):.4f}")

    # ===================================================================
    # APARTADO 4: Comparación de tiempos de ejecución
    # ===================================================================
    print("\n" + "=" * 60)
    print("APARTADO 4: Tiempos de ejecución")
    print("=" * 60)

    sigma_u, sigma_v = 1.0, 1.0
    x_true_t, y_obs_t = simulate_system(A, B, sigma_u, sigma_v, T)

    n_timing_runs = 20
    N_particles_timing = [50, 100, 200, 500, 1000, 2000]

    # Tiempo Kalman
    times_kalman = []
    for _ in range(n_timing_runs):
        t0 = time.perf_counter()
        kalman_filter(y_obs_t, A, B, sigma_u, sigma_v)
        times_kalman.append(time.perf_counter() - t0)
    mean_time_kalman = np.mean(times_kalman)
    print(f"  Kalman:  {mean_time_kalman * 1000:.3f} ms  (promedio de {n_timing_runs} ejecuciones)")

    # Tiempo Filtro de partículas
    times_pf = {}
    for N_p in N_particles_timing:
        runs = []
        for _ in range(n_timing_runs):
            t0 = time.perf_counter()
            particle_filter(y_obs_t, A, B, sigma_u, sigma_v, N_p)
            runs.append(time.perf_counter() - t0)
        times_pf[N_p] = np.mean(runs)
        print(f"  PF (N={N_p:>4d}): {times_pf[N_p] * 1000:.3f} ms")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(['Kalman'] + [f'PF N={N}' for N in N_particles_timing],
           [mean_time_kalman * 1000] + [times_pf[N] * 1000 for N in N_particles_timing],
           color=['steelblue'] + ['indianred'] * len(N_particles_timing))
    ax.set_ylabel('Tiempo (ms)')
    ax.set_title('Tiempo de ejecución: Kalman vs Filtro de Partículas')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig('fig5_tiempos_ejecucion.png', dpi=150)
    plt.close()
    print("  → fig5_tiempos_ejecucion.png generada.")

    # ===================================================================
    # APARTADO 5: Varianza de z_100 = ||x_100|| condicionada a y_{1:100}
    # ===================================================================
    print("\n" + "=" * 60)
    print("APARTADO 5: Estimación de Var(z_100 | y_{1:100})")
    print("=" * 60)

    sigma_u, sigma_v = 1.0, 1.0
    N_p_var = 5000  # Más partículas para mejor estimación de la varianza
    x_true_v, y_obs_v = simulate_system(A, B, sigma_u, sigma_v, T)

    var_z, mean_z, z_parts, w_final = estimate_variance_z100(
        y_obs_v, A, B, sigma_u, sigma_v, N_p_var
    )

    z_true = np.sqrt(x_true_v[:, T] @ x_true_v[:, T])
    print(f"  z_100 real:             {z_true:.4f}")
    print(f"  E[z_100 | y_{{1:100}}]:   {mean_z:.4f}")
    print(f"  Var[z_100 | y_{{1:100}}]: {var_z:.6f}")
    print(f"  Std[z_100 | y_{{1:100}}]: {np.sqrt(var_z):.4f}")
    print(f"  Nº partículas:          {N_p_var}")

    print("\n  Método utilizado:")
    print("  Se ejecuta el filtro de partículas hasta t=100, obteniendo partículas")
    print("  {x_100^(i)} con pesos {w^(i)}. Se calcula z^(i) = ||x_100^(i)|| para")
    print("  cada partícula. La media y la varianza se estiman como:")
    print("    E[z] ≈ Σ w^(i) z^(i)")
    print("    Var[z] ≈ Σ w^(i) (z^(i) - E[z])²")

    # Histograma de z_100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z_parts, bins=60, weights=w_final, density=True,
            color='steelblue', alpha=0.7, edgecolor='white', label='Distribución de $z_{100}$')
    ax.axvline(z_true, color='k', linewidth=2, linestyle='-', label=f'$z_{{100}}$ real = {z_true:.2f}')
    ax.axvline(mean_z, color='r', linewidth=2, linestyle='--', label=f'Media estimada = {mean_z:.2f}')
    ax.set_xlabel('$z_{100} = \\|x_{100}\\|$')
    ax.set_ylabel('Densidad')
    ax.set_title(f'Distribución posterior de $z_{{100}}$ | $y_{{1:100}}$ — Var = {var_z:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fig6_varianza_z100.png', dpi=150)
    plt.close()
    print("  → fig6_varianza_z100.png generada.")


# ===========================================================================
# Ejecución principal
# ===========================================================================

if __name__ == '__main__':
    run_all_experiments()