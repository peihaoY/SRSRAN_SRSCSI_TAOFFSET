import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import deque

data_lock = threading.Lock()

latest_csi_data = {}
ta_history = deque()

fig = None
axs = {}
lines = {}
rx_ports = set()

last_phase = {}


def recv_csi_sctp(listen_ip="0.0.0.0", listen_port=5000):
    global latest_csi_data, rx_ports, ta_history
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_SCTP)
    sock.bind((listen_ip, listen_port))
    sock.listen(1)

    print(f"Listening on {listen_ip}:{listen_port} ...")
    conn, addr = sock.accept()
    print(f"Connected by {addr}")

    while True:
        data = conn.recv(8192) 
        if not data:
            break

        floats = np.frombuffer(data, dtype=np.float32)

        if len(floats) < 3:
            continue

        rx_port = int(floats[0])
        tx_port = int(floats[1])
        time_alignment_s = floats[2]

        csi_complex = floats[3:].reshape(-1, 2)
        csi = csi_complex[:, 0] + 1j * csi_complex[:, 1]

        with data_lock:
            latest_csi_data[rx_port] = csi
            ta_history.append(time_alignment_s * 1e6)
            rx_ports.add(rx_port)

        print(f"RX={rx_port}, TX={tx_port}, TA={time_alignment_s * 1e6:.3f} µs, CSI_len={len(csi)}")

    conn.close()
    sock.close()


def setup_plots():
    global fig, axs, lines

    while not rx_ports:
        #print("Waiting for first data packet to setup plots...")
        threading.Event().wait(0.5)
    
    n_rx = len(rx_ports)
    rx_port_list = sorted(list(rx_ports))
    
    print(f"Setting up plots for RX ports: {rx_port_list}")

    fig = plt.figure(figsize=(4 * (n_rx + 1), 8))
    gs = fig.add_gridspec(2, n_rx + 1)

    for i, rx_port in enumerate(rx_port_list):
        ax_mag = fig.add_subplot(gs[0, i])
        l1, = ax_mag.plot([], [], label=f'RX {rx_port}')
        ax_mag.set_title(f"RX {rx_port} - Magnitude", fontsize=14)
        ax_mag.set_ylabel('Magnitude', fontsize=12)
        ax_mag.set_xlabel('Subcarrier Index', fontsize=12)
        ax_mag.grid(True)
        ax_mag.set_ylim(0, 1)

        ax_phase = fig.add_subplot(gs[1, i])
        l2, = ax_phase.plot([], [], label=f'RX {rx_port}')
        ax_phase.set_title(f"RX {rx_port} - Phase", fontsize=14)
        ax_phase.set_ylabel('Unwrapped Phase (rad)', fontsize=12)
        ax_phase.set_xlabel('Subcarrier Index', fontsize=12)
        ax_phase.grid(True)
        ax_phase.set_ylim(-10, 10)

        axs[rx_port] = {'mag': ax_mag, 'phase': ax_phase}
        lines[rx_port] = {'mag': l1, 'phase': l2}

    ax_ta = fig.add_subplot(gs[:, n_rx])
    l_ta, = ax_ta.plot([], [], 'r.-', label='Time Alignment') # Red line with dots
    ax_ta.set_title("Time Alignment History", fontsize=14)
    ax_ta.set_ylabel("Time Alignment (µs)", fontsize=12)
    ax_ta.set_xlabel("Sample Index", fontsize=12)
    ax_ta.grid(True)
    ax_ta.legend()

    axs['ta'] = ax_ta
    lines['ta'] = l_ta

    plt.tight_layout(pad=2.0)


def update_plots(frame):
    """
    This function is called repeatedly by FuncAnimation to update plot data.
    """
    global latest_csi_data, ta_history, lines, axs
    
    updated_lines = []

    with data_lock:
        local_csi_data = latest_csi_data.copy()
        local_ta_history = list(ta_history)

    if not local_csi_data or not lines:
        return []

    # --- Update CSI Plots ---
    """
    for rx_port, csi_data in local_csi_data.items():
        if rx_port not in lines:
            continue
            
        mag = np.abs(csi_data)
        phase_unwrapped = np.unwrap(np.angle(csi_data))
        
        x_data = np.arange(len(mag))
        
        # Update magnitude data and limits
        lines[rx_port]['mag'].set_data(x_data, mag)
        axs[rx_port]['mag'].set_xlim(0, len(mag) - 1 if len(mag) > 1 else 1)
        
        if len(csi_data) > 1:
            k = np.arange(len(csi_data))
            p = np.polyfit(k, phase_unwrapped, 1)
            compensated_phase = phase_unwrapped - (p[0] * k + p[1])
        else:
            compensated_phase = phase_unwrapped
        lines[rx_port]['phase'].set_data(x_data, phase_unwrapped)

        # Update phase data and limits
        # lines[rx_port]['phase'].set_data(x_data, phase_unwrapped)
        axs[rx_port]['phase'].set_xlim(0, len(phase_unwrapped) - 1 if len(phase_unwrapped) > 1 else 1)
        
        updated_lines.extend([lines[rx_port]['mag'], lines[rx_port]['phase']])
    """
    for rx_port, csi_data in local_csi_data.items():
        if rx_port not in lines:
            continue

        mag = np.abs(csi_data)
        phase_raw = np.unwrap(np.angle(csi_data))

        mean_phase = np.angle(np.mean(np.exp(1j * phase_raw)))
        phase_centered = phase_raw - mean_phase

        x = np.arange(len(phase_centered))
        slope, intercept = np.polyfit(x, phase_centered, 1)
        phase_detrended = phase_centered - (slope * x + intercept)

        phase_to_plot = phase_detrended   

        x_data = np.arange(len(mag))
        lines[rx_port]['mag'].set_data(x_data, mag)
        axs[rx_port]['mag'].set_xlim(0, len(mag) - 1 if len(mag) > 1 else 1)

        lines[rx_port]['phase'].set_data(x_data, phase_to_plot)
        axs[rx_port]['phase'].set_xlim(0, len(phase_to_plot) - 1 if len(phase_to_plot) > 1 else 1)

        updated_lines.extend([lines[rx_port]['mag'], lines[rx_port]['phase']])

    if 'ta' in lines and local_ta_history:
        ta_line = lines['ta']
        ta_ax = axs['ta']
        x_ta_data = np.arange(len(local_ta_history))
        ta_line.set_data(x_ta_data, local_ta_history)
        ta_ax.relim()
        ta_ax.autoscale_view(True, True, True)
        updated_lines.append(ta_line)

    return updated_lines


if __name__ == "__main__":
    t = threading.Thread(target=recv_csi_sctp, args=("0.0.0.0", 5000), daemon=True)
    t.start()
    setup_plots()
    ani = FuncAnimation(fig, update_plots, interval=50, blit=False)
    plt.show()
