//run with cargo run > ../results/solution_fem_1d.csv

use std::f64::consts::PI;

fn main() {
    // Parameters
    let nx = 100; // Number of spatial points
    let nt = 5000; // Number of time steps
    let x_min = 0.0;
    let x_max = 1.0;
    let t_max = 1.0;
    let nu = 0.01 / PI; // Viscosity coefficient

    // Derived quantities
    let dx = (x_max - x_min) / (nx - 1) as f64;
    let dt = t_max / nt as f64;

    // Ensure the CFL condition for numerical stability
    let max_u = 1.0; // Maximum |u| from the initial condition (-sin(pi x))
    if dt > dx.min(dx * dx / nu) / max_u {
        panic!("CFL condition not satisfied: reduce dt or increase nx");
    }

    // Initialize grid and initial condition
    let mut x: Vec<f64> = (0..nx).map(|i| x_min + i as f64 * dx).collect();
    let mut u: Vec<f64> = x.iter().map(|&xi| -f64::sin(PI * xi)).collect();
    let mut u_new = u.clone();

    // Time-stepping loop
    for _ in 0..nt {
        for i in 1..nx - 1 {
            // Burgers' equation discretization
            let du_dx = (u[i + 1] - u[i - 1]) / (2.0 * dx); // Central difference for du/dx
            let d2u_dx2 = (u[i + 1] - 2.0 * u[i] + u[i - 1]) / (dx * dx); // Central difference for d2u/dx2

            // Update u_new using forward Euler
            u_new[i] = u[i] - u[i] * du_dx * dt + nu * d2u_dx2 * dt;
        }

        // Boundary conditions
        u_new[0] = 0.0;
        u_new[nx - 1] = 0.0;

        // Swap the new solution into the old
        u.copy_from_slice(&u_new);
    }

    // Print the final solution
    println!("x,u");
    for i in 0..nx {
        println!("{},{}", x[i], u[i]);
    }
}