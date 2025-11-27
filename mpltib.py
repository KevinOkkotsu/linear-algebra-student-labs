import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def test_matplotlib(save_path="simple_plot.png", show_if_display=True):
    # Print versions and backend for quick debug
    print("numpy:", np.__version__)
    print("matplotlib:", matplotlib.__version__)
    print("matplotlib backend:", matplotlib.get_backend())

    # Create a simple plot
    x = np.linspace(0, 2 * np.pi, 300)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y1, label="sin(x)")
    plt.plot(x, y2, label="cos(x)")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.title("Simple sin / cos plot")
    plt.legend()
    plt.grid(True)

    # Save always so you can inspect the file even in headless envs
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")

    # Show interactively only if a display is available (or on Windows)
    if show_if_display and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name == "nt"):
        print("Displaying plot interactively...")
        plt.show()
    else:
        print("No display detected or show suppressed.")
        print("To view the saved image locally you can run:")
        print(f"  xdg-open {save_path}   # on Linux with GUI")
        print("Or run with a virtual X server:")
        print(f"  xvfb-run -a python3 {os.path.basename(__file__)}")

if __name__ == "__main__":
    test_matplotlib()