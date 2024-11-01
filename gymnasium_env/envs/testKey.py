from pynput import keyboard

def on_press(key):
    try:
        if key.char == 'q':  # 检测按下 'q' 键
            print("You pressed 'q'. Exiting...")
            return False  # 返回 False 以停止监听
    except AttributeError:
        pass

print("Press 'q' to quit")

# 启动键盘监听器
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

