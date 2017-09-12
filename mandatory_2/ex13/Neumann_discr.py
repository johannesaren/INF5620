import sympy as sym
t,x = sym.symbols('t x')

def solver(I,V,f,q,U_0,U_L,L,dt,C,T,
            user_action=None, version='scalar', task ='task_a',
            stability_safety_factor=1.0):
    """Solve u_tt = (c**2*u_x)_x + f on (0),L)X(0,T].)"""

    Nt = int(round(T/dt))
    t = linspace(0, Nt*dt, Nt+1)      # Mesh points in time

    # fixes if the q is either a number or a vector:
    if isinstance(q, (float,int)):
        q_max = q
    elif callable(q):
        q_max = max([q(x_) for x_ in linspace(0,L,101)])

    dx = dt*q_max/(stability_safety_factor*C)
    Nx = int(round(L/dx))
    x = linspace(0, L, Nx+1)          # Mesh points in space

    # and again,fixes if the q is either a number or a vector:
    if isinstance(q, (float,int)):
        q = q*ones(len(x))
    elif callable(q):
        q_ = zeros(len(x))
        for i in range(Nx+1):
            q_[i] = q(x[i])
        q = q_

    # SHORTENING VARIABLES FOR LATER
    C2 = (dt/dx)**2
    dt2 = dt*dt

    # WRAPS THE INPUT "f, I, V, U_0, U_L" if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: zeros(x.shape)

    u   = zeros(Nx+1)   # Solution array at new time level
    u_1 = zeros(Nx+1)   # Solution at 1 time level back
    u_2 = zeros(Nx+1)   # Solution at 2 time levels back

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # LOADING INITIAL CONDITIONS INTO u_1
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])
    if user_action is not None:
        user_action(u_1, x, t, 0, dx)

    # FORMULA FOR THE FIRST STEP, different from the rest.
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    # BOUNDARY CONDITIONS FIRST STEP
    i = Ix[0]
    if U_0 is None:
        # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        u[i] = u_1[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - \
               0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
               0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_0(dt)

    i = Ix[-1] # BOUNDARY CONDITIONS FOR "THE OTHER SIDE"
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        u[i] = u_1[i] + dt*V(x[i]) + \
               0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i]) - \
               0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + \
               0.5*dt2*f(x[i], t[0])
    else:
        u[i] = U_L(dt)

    if user_action is not None:
        user_action(u, x, t, 1, dx)

    # UPDATING SOLVERS BEFORE THE NEXT STEP
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # UPDATING ALL INNER POINTS
        u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + \
        C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) -
            0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + \
            dt2*f(x[1:-1], t[n])

    # INSERTING BOUNDARY CONDITIONS for x=0
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0

        #-------------------------------------------------
        # THIS IF-test SETS THE DIFFERENT CONDITIONS FOR
        # "Exercise 13 a)", "Exercise 13 b)", etc.
        if task == 'task_a':
            # INSERTING BOUNDARY CONDITIONS for x=START
            i = Ix[0]
            ip1 = i+1
            im1 = ip1
            # USING THE APPROXIMATION (54)
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*2*q[i]*(u_1[ip1] - u_1[i])  \
                 + dt2*f(x[i], t[n])
            # INSERTING BOUNDARY CONDITIONS for x=END
            i = Ix[-1]
            im1 = i-1
            ip1 = im1
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*2*q[i]*(u_1[ip1] - u_1[i])  \
                    + dt2*f(x[i], t[n])
        elif task == 'task_b':
            # INSERTING BOUNDARY CONDITIONS for x=START
            i = Ix[0]
            ip1 = i+1
            im1 = ip1
            # ASSUMING dq/dx = 0, therefore:
            # USING THE SCHEME FOR (57)
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*(q[i] + q[im1])*(u_1[im1] - u_1[i]) + \
                dt2*f(x[i], t[n])
            # INSERTING BOUNDARY CONDITIONS for x=END
            i = Ix[-1]
            im1 = i-1
            ip1 = im1
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*(q[i] + q[ip1])*(u_1[ip1] - u_1[i]) + \
                dt2*f(x[i], t[n])
        elif task == 'task_c':
            # INSERTING BOUNDARY CONDITIONS for x=START
            i = Ix[0]
            # USING THE THIRD DISCRETIZATION.
            # LESS ACCURATE
            u[i] = u[i+1]
            # INSERTING BOUNDARY CONDITIONS for x=END
            i = Ix[-1]
            u[i] = u[i-1]
        elif task == 'task_d':
            # INSERTING BOUNDARY CONDITIONS for x=START
            i = Ix[0]
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) + \
                dt2*f(x[i], t[n])
            # INSERTING BOUNDARY CONDITIONS for x=END
            i = Ix[-1]
            u[i] = - u_2[i] + 2*u_1[i] + \
                C2*0.5*(q[i] + q[i-1])*(u_1[i-1] - u_1[i]) + \
                dt2*f(x[i], t[n])

        if user_action is not None:
            if user_action(u, x, t, n+1, dx):
                break
    # UPDATING DATA STRUCTURES FOR THE NEXT STEP
        u_2, u_1, u = u_1, u, u_2
    u = u_1
    return u, x, t

# FINDING u_exact

from numpy import *

def u_exact(L = 4*pi, omega = 1):
    u_e = lambda x,t : sym.cos(sym.pi*x/L)*sym.cos(omega*t)
    return u_e


def find_source_term(u,q):  # DIFFERENCIATING:
    return sym.diff(u(x,t),t,2) - sym.diff(q(x)*sym.diff(u(x,t),x),x)

def error_func(u,x,t,n,dx): # DEFINING THE ERRORFUNCTION
    u_e = linspace(0,0,len(x))
    u_e1 = u_exact(L = 2.0)
    for i in range(len(x)):
        u_e[i] = u_e1(x[i],t[n])
    diff = u_e - u
    #print diff, dx
    error = sym.sqrt(dx*sum(diff**2))
    return error

# DEFINING THE ERRORFUNCTION
def convergence_rates(error, error_1, h, h_half):
    return sym.log(error_1/error)/ sym.log(h_half/h)

class Action:
    """Stores last solution"""
    def __call__(self, u, x, t, n, dx):
        if n == len(t)-1:
            self.u = u.copy()
            self.x = x.copy()
            self.t = t
            self.n = n
            self.dx = dx

#--------------------------------------------------------------

def main_a():
    # DEFINING THE INPUT WE WISH FOR EACH TASK
    L = 2.0
    Nx = 20
    T = 2
    C = 0.5
    dt = C*(L/Nx)/6 # EXPERIMENTING WITH DIFFERENT dt's
    safety_fac = 1

    u_e = u_exact(L)

    I = lambda x: u_e(x,0)
    q = lambda x: 1 + (x - L/2)**4 # SETING UP THE CORRECT q
    V = lambda x: 0
    source = find_source_term(u_e,q)
    f = sym.lambdify((x,t),source)

    dt_1 = Action()

    solver(I,V,f,q,None,None,L,dt,C,T,
        user_action=dt_1, version='scalar', task ='task_a',
        stability_safety_factor=safety_fac)

    dt_2 = Action()

    solver(I,V,f,q,None,None,L,dt/2,C,T,
        user_action=dt_2, version='scalar', task ='task_a',
        stability_safety_factor=safety_fac)

    e = error_func(dt_1.u, dt_1.x, dt_1.t, dt_1.n, dt_1.dx)
    e_1 = error_func(dt_2.u, dt_2.x, dt_2.t, dt_2.n, dt_2.dx)

    r = convergence_rates(e,e_1,dt,dt/2)
    print 'The convergence rate is:', r

def main_b():
    L = 2.0
    Nx = 20
    T = 2
    C = 0.5
    dt = C*(L/Nx)/3
    safety_fac = 1

    u_e = u_exact(L)

    I = lambda x: u_e(x,0)
    q = lambda x: 1 + sym.cos((sym.pi*x)/L) # EDITING, so we get the right q
    V = lambda x: 0
    source = find_source_term(u_e,q)
    f = sym.lambdify((x,t),source)

    dt_1 = Action()

    solver(I,V,f,q,None,None,L,dt,C,T,
        user_action=dt_1, version='scalar', task ='task_b',
        stability_safety_factor=safety_fac)

    dt_2 = Action()

    solver(I,V,f,q,None,None,L,dt/2,C,T,
        user_action=dt_2, version='scalar', task ='task_b',
        stability_safety_factor=safety_fac)

    e = error_func(dt_1.u, dt_1.x, dt_1.t, dt_1.n, dt_1.dx)
    e_1 = error_func(dt_2.u, dt_2.x, dt_2.t, dt_2.n, dt_2.dx)

    r = convergence_rates(e,e_1,dt,dt/2)
    print 'The convergence rate is:', r

def main_c():
    L = 2.0
    Nx = 20
    T = 2
    C = 0.5
    dt = C*(L/Nx)/6
    safety_fac = 1

    u_e = u_exact(L)

    I = lambda x: u_e(x,0)
    q_1 = lambda x: 1 + (x - L/2)**4            # SETTING UP THE q FROM task a)
    q_2 = lambda x: 1 + sym.cos((sym.pi*x)/L)   # SETTING UP THE q FROM task b)
    V = lambda x: 0
    q = [q_1, q_2]  # STORING THEM IN AN ARRAY, to call them in the for loop

    for i in range (0,2):
        source = find_source_term(u_e,q[i])
        f = sym.lambdify((x,t),source)

        dt_1 = Action()

        solver(I,V,f,q[i],None,None,L,dt,C,T,
            user_action=dt_1, version='scalar', task ='task_c',
            stability_safety_factor=safety_fac)

        dt_2 = Action()

        solver(I,V,f,q[i],None,None,L,dt/2,C,T,
            user_action=dt_2, version='scalar', task ='task_c',
            stability_safety_factor=safety_fac)

        e = error_func(dt_1.u, dt_1.x, dt_1.t, dt_1.n, dt_1.dx)
        e_1 = error_func(dt_2.u, dt_2.x, dt_2.t, dt_2.n, dt_2.dx)

        r = convergence_rates(e,e_1,dt,dt/2)
        print 'The convergence rate is:', r

def main_d():
    L = 2.0
    Nx = 20
    T = 2
    C = 0.5
    dt = C*(L/Nx)/6
    safety_fac = 1

    u_e = u_exact(L)

    I = lambda x: u_e(x,0)
    V = lambda x: 0
    q_1 = lambda x: 1 + (x - L/2)**4
    q_2 = lambda x: 1 + sym.cos((sym.pi*x)/L)
    q = [q_1, q_2]
    source_1 = find_source_term(u_e,q_1)
    source_2 = find_source_term(u_e,q_2)
    f_1 = sym.lambdify((x,t),source_1)
    f_2 = sym.lambdify((x,t), source_2)
    f = [f_1, f_2]


    for i in range (0,2):
        #source = find_source_term(u_e,q[i])
        #f = sym.lambdify((x,t),source)

        dt_1 = Action()

        solver(I,V,f[i],q[i],None,None,L,dt,C,T,
            user_action=dt_1, version='scalar', task ='task_d',
            stability_safety_factor=safety_fac)

        dt_2 = Action()

        solver(I,V,f[i],q[i],None,None,L,dt/2,C,T,
            user_action=dt_2, version='scalar', task ='task_d',
            stability_safety_factor=safety_fac)

        e = error_func(dt_1.u, dt_1.x, dt_1.t, dt_1.n, dt_1.dx)
        e_1 = error_func(dt_2.u, dt_2.x, dt_2.t, dt_2.n, dt_2.dx)

        r = convergence_rates(e,e_1,dt,dt/2)
        print 'The convergence rate is:', r

if __name__ == '__main__':
     main_a()
     #main_b()
     #main_c()
     #main_d()
