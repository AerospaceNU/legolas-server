from gimbal.ronin_controller import RoninController
import time

def test():
    ronin = RoninController("can0")

    # ronin.reset_to_zero()

    ronin.set_pitch_position(200)
    ronin.set_yaw_position(120)
    ronin.set_roll_position(20)
    # ronin.get_current_position()

    print("RONIN: Jetson mode =", ronin.jetson)
 
    print(ronin.get_current_position())

    print("\nDone!")
    if ronin.jetson and hasattr(ronin, '_bus'):
        ronin._bus.shutdown()


if __name__ == "__main__":
    test()


