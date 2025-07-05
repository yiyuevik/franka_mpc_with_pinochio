/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_franka_14dof_H_
#define ACADOS_SOLVER_franka_14dof_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define FRANKA_14DOF_NX     14
#define FRANKA_14DOF_NZ     0
#define FRANKA_14DOF_NU     7
#define FRANKA_14DOF_NP     0
#define FRANKA_14DOF_NP_GLOBAL     0
#define FRANKA_14DOF_NBX    0
#define FRANKA_14DOF_NBX0   14
#define FRANKA_14DOF_NBU    0
#define FRANKA_14DOF_NSBX   0
#define FRANKA_14DOF_NSBU   0
#define FRANKA_14DOF_NSH    0
#define FRANKA_14DOF_NSH0   0
#define FRANKA_14DOF_NSG    0
#define FRANKA_14DOF_NSPHI  0
#define FRANKA_14DOF_NSHN   0
#define FRANKA_14DOF_NSGN   0
#define FRANKA_14DOF_NSPHIN 0
#define FRANKA_14DOF_NSPHI0 0
#define FRANKA_14DOF_NSBXN  0
#define FRANKA_14DOF_NS     0
#define FRANKA_14DOF_NS0    0
#define FRANKA_14DOF_NSN    0
#define FRANKA_14DOF_NG     0
#define FRANKA_14DOF_NBXN   0
#define FRANKA_14DOF_NGN    0
#define FRANKA_14DOF_NY0    10
#define FRANKA_14DOF_NY     10
#define FRANKA_14DOF_NYN    3
#define FRANKA_14DOF_N      128
#define FRANKA_14DOF_NH     0
#define FRANKA_14DOF_NHN    0
#define FRANKA_14DOF_NH0    0
#define FRANKA_14DOF_NPHI0  0
#define FRANKA_14DOF_NPHI   0
#define FRANKA_14DOF_NPHIN  0
#define FRANKA_14DOF_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct franka_14dof_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *impl_dae_fun;
    external_function_external_param_casadi *impl_dae_fun_jac_x_xdot_z;
    external_function_external_param_casadi *impl_dae_jac_x_xdot_u_z;




    // cost

    external_function_external_param_casadi *cost_y_fun;
    external_function_external_param_casadi *cost_y_fun_jac_ut_xt;



    external_function_external_param_casadi cost_y_0_fun;
    external_function_external_param_casadi cost_y_0_fun_jac_ut_xt;



    external_function_external_param_casadi cost_y_e_fun;
    external_function_external_param_casadi cost_y_e_fun_jac_ut_xt;


    // constraints







} franka_14dof_solver_capsule;

ACADOS_SYMBOL_EXPORT franka_14dof_solver_capsule * franka_14dof_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_free_capsule(franka_14dof_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int franka_14dof_acados_create(franka_14dof_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int franka_14dof_acados_reset(franka_14dof_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of franka_14dof_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_create_with_discretization(franka_14dof_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_update_time_steps(franka_14dof_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_update_qp_solver_cond_N(franka_14dof_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_update_params(franka_14dof_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_update_params_sparse(franka_14dof_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_set_p_global_and_precompute_dependencies(franka_14dof_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int franka_14dof_acados_solve(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_setup_qp_matrices_and_factorize(franka_14dof_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int franka_14dof_acados_free(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void franka_14dof_acados_print_stats(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int franka_14dof_acados_custom_update(franka_14dof_solver_capsule* capsule, double* data, int data_len);


ACADOS_SYMBOL_EXPORT ocp_nlp_in *franka_14dof_acados_get_nlp_in(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *franka_14dof_acados_get_nlp_out(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *franka_14dof_acados_get_sens_out(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *franka_14dof_acados_get_nlp_solver(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *franka_14dof_acados_get_nlp_config(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *franka_14dof_acados_get_nlp_opts(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *franka_14dof_acados_get_nlp_dims(franka_14dof_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *franka_14dof_acados_get_nlp_plan(franka_14dof_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_franka_14dof_H_
