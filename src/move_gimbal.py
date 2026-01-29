from gimbal.ronin_controller import RoninController
import time

def test():
    ronin = RoninController("can0")

    ronin.set_pitch_position(0)
    ronin.set_yaw_position(0)
    ronin.set_roll_position(0)
    ronin.get_current_position()

    print("RONIN: Jetson mode =", ronin.jetson)
 
    print(ronin.get_current_position())

    print("\nDone!")
    if ronin.jetson and hasattr(ronin, '_bus'):
        ronin._bus.shutdown()


if __name__ == "__main__":
    test()


