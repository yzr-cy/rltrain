# This code is implemented by Chauby (chaubyZou@163.com), feel free to use it.
# the repository of this code is: https://github.com/chauby/ZMP_preview_control.git


#%% Initialize global variations
import numpy as np
from control.matlab import dare # for solving the discrete algebraic Riccati equation

# ------------------------------- Preview Control, reference: 2003-Biped Walking Pattern Generation by Using Preview Control of Zero-moment Point
def calculatePreviewControlParams(A, B, C, Q, R, N):
    [P, _, _] = dare(A, B, C.T*Q*C, R)
    K = (R + B.T*P*B).I*(B.T*P*A)

    f = np.zeros((1, N))
    for i in range(N):
        f[0,i] = (R+B.T*P*B).I*B.T*(((A-B*K).T)**i)*C.T*Q

    return K, f

# ------------------------------- Improved Preview Control, reference: 2007-General ZMP Preview Control for Bipedal Walking
def calculatePreviewControlParams2(A, B, C, Q, R, N):
    C_dot_A = C*A
    C_dot_B = C*B

    A_tilde = np.matrix([[1, C_dot_A[0,0], C_dot_A[0,1], C_dot_A[0,2]],
                            [0, A[0,0], A[0,1], A[0,2]],
                            [0, A[1,0], A[1,1], A[1,2]],
                            [0, A[2,0], A[2,1], A[2,2]]])
    B_tilde = np.matrix([[C_dot_B[0,0]],
                            [B[0,0]],
                            [B[1,0]],
                            [B[2,0]]])
    C_tilde = np.matrix([[1, 0, 0, 0]])

    [P_tilde, _, _] = dare(A_tilde, B_tilde, C_tilde.T*Q*C_tilde, R)
    K_tilde = (R + B_tilde.T*P_tilde*B_tilde).I*(B_tilde.T*P_tilde*A_tilde)

    Ks = K_tilde[0, 0]
    Kx = K_tilde[0, 1:]

    Ac_tilde = A_tilde - B_tilde*K_tilde

    G = np.zeros((1, N))

    G[0] = -Ks
    I_tilde = np.matrix([[1],[0],[0],[0]])
    X_tilde = -Ac_tilde.T*P_tilde*I_tilde

    for i in range(N):
        G[0,i] = (R + B_tilde.T*P_tilde*B_tilde).I*(B_tilde.T)*X_tilde
        X_tilde = Ac_tilde.T*X_tilde

    return Ks, Kx, G



# %% demo
if __name__ == '__main__':
  
    step_pos = []
    z_c = 0.93
    g = 9.81
    dt = 0.06
    t_step = 0.3 # timing for one step
    t_preview = 2.5 # timing for preview

    vx = 0
    vx_max = 1
    acc_time = 3 # 加速时间
    ave_time = 100
    y_flag = 1
    step_pos.append([0,0])
    maxStep = int((acc_time+ave_time)/dt)
    for i in np.arange(0,maxStep,1):
        step_pos.append([step_pos[i][0] + vx * t_step, 0.14*y_flag])

        y_flag = y_flag*(-1)
        if vx < vx_max:
            vx = vx + vx_max/acc_time*dt
        print(vx)
    step_pos = np.array(step_pos)

    n_step = len(step_pos)
    t_simulation = n_step*t_step - t_preview - dt # timing for simulation

    N_preview = int(t_preview/dt) # preview length
    N_simulation = int(t_simulation/dt)

    # Generate ZMP trajectory
    ZMP_x_ref = []
    ZMP_y_ref = []

    for i in np.arange(0,n_step,1):
        for j in np.arange(0,5,1):
            ZMP_x_ref.append(step_pos[i,0])
            ZMP_y_ref.append(step_pos[i,1])

    # Define basic matrix
    A = np.mat(([1, dt, dt**2/2],
                [0, 1, dt],
                [0, 0, 1]))
    B = np.mat((dt**3/6, dt**2/2, dt)).T
    C = np.mat((1, 0, -z_c/g))

    Q = 10
    R = 1e-6

    # Calculate Preview control parameters
    K, f = calculatePreviewControlParams(A, B, C, Q, R, N_preview)

    # Calculate Improved Preview control parameters
    Ks, Kx, G = calculatePreviewControlParams2(A, B, C, Q, R, N_preview)

    # ------------------------------- for Improved Preview Control 2
    ux_3 = np.asmatrix(np.zeros((N_simulation, 1)))
    uy_3 = np.asmatrix(np.zeros((N_simulation, 1)))
    COM_x_3 = np.asmatrix(np.zeros((3, N_simulation+1)))
    COM_y_3 = np.asmatrix(np.zeros((3, N_simulation+1)))

    # record data for plot
    COM_x_record_3 = []
    COM_y_record_3 = []
    ZMP_x_record_3 = []
    ZMP_y_record_3 = []
    VEL_x_record_3 = []
    VEL_y_record_3 = []


    e_x_3 = np.zeros((N_simulation, 1))
    e_y_3 = np.zeros((N_simulation, 1))

    sum_e_x = 0
    sum_e_y = 0


    # main loop
    for k in range(N_simulation):
        ZMP_x_preview = np.asmatrix(ZMP_x_ref[k:k+N_preview]).T
        ZMP_y_preview = np.asmatrix(ZMP_y_ref[k:k+N_preview]).T


        # --------------------------------- 3: Improved Preview Control with the summary of history errors
        # update ZMP
        ZMP_x = C*COM_x_3[:,k]
        ZMP_y = C*COM_y_3[:,k]
        ZMP_x_record_3.append(ZMP_x[0,0])
        ZMP_y_record_3.append(ZMP_y[0,0])

        # calculate errors
        e_x_3[k] = ZMP_x - ZMP_x_ref[k] 
        e_y_3[k] = ZMP_y - ZMP_y_ref[k]
        sum_e_x += e_x_3[k]
        sum_e_y += e_y_3[k]

        # update u
        ux_3[k] = -Ks*sum_e_x - Kx*COM_x_3[:, k] - G*ZMP_x_preview
        uy_3[k] = -Ks*sum_e_y - Kx*COM_y_3[:, k] - G*ZMP_y_preview

        # update COM state
        COM_x_3[:,k+1] = A*COM_x_3[:, k] + B*ux_3[k]
        COM_y_3[:,k+1] = A*COM_y_3[:, k] + B*uy_3[k]
        COM_x_record_3.append(COM_x_3[0,k])
        COM_y_record_3.append(COM_y_3[0,k])
        VEL_x_record_3.append(COM_x_3[1,k])
        VEL_y_record_3.append(COM_y_3[1,k])


    import pandas as pd

    min_length = min(len(COM_x_record_3), len(COM_y_record_3), len(VEL_x_record_3), len(VEL_y_record_3))

    # 截取 ZMP_x_ref 和 ZMP_y_ref 到相同的长度
    ZMP_x_ref = ZMP_x_ref[:min_length]
    ZMP_y_ref = ZMP_y_ref[:min_length]
    COM_x_record_3 = COM_x_record_3[:min_length]
    COM_y_record_3 = COM_y_record_3[:min_length]
    VEL_x_record_3 = VEL_x_record_3[:min_length]
    VEL_y_record_3 = VEL_y_record_3[:min_length]


    # 创建一个字典，将每个记录列表作为字典中的列
    data = {
        'COM_x': COM_x_record_3,
        'COM_y': COM_y_record_3,
        'VEL_x': VEL_x_record_3,
        'VEL_y': VEL_y_record_3,
        'ZMP_REF_x': ZMP_x_ref,
        'ZMP_REF_y': ZMP_y_ref,
    }

    # 将字典转换为 DataFrame
    df = pd.DataFrame(data)

    # 将 DataFrame 保存到 CSV 文件
    df.to_csv("ZMP_preview_control_results.csv", index=False)



    # plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Preview Control 3")
    plt.subplot(3,1,1)
    plt.plot(ZMP_x_ref, ZMP_y_ref, 'g--', label='ZMP_ref')
    plt.plot(ZMP_x_record_3, ZMP_y_record_3, 'b', label='ZMP')
    plt.plot(COM_x_record_3, COM_y_record_3, 'r--', label='COM')
    plt.legend()
    plt.subplot(3,1,2)
    # plt.plot(ZMP_x_ref, ZMP_y_ref, 'g--', label='ZMP_ref')
    # plt.plot(ZMP_x_record_3, ZMP_y_record_3, 'b', label='ZMP')
    plt.plot(VEL_x_record_3, 'r--', label='VEL')
    plt.legend()
    plt.subplot(3,1,3)
    # plt.plot(ZMP_x_ref, ZMP_y_ref, 'g--', label='ZMP_ref')
    # plt.plot(ZMP_x_record_3, ZMP_y_record_3, 'b', label='ZMP')
    plt.plot(VEL_y_record_3, 'r--', label='VEL')
    plt.legend()

    plt.show()

