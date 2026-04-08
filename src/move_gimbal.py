from gimbal.ronin_controller import RoninController


def test():
    ronin = RoninController("can0")

    ronin.set_pitch_position(200)
    ronin.set_yaw_position(120)

    print("RONIN: Jetson mode =", ronin.jetson)

    print("\nDone!")
    if ronin.jetson and hasattr(ronin, '_bus'):
        ronin._bus.shutdown()


if __name__ == "__main__":
    test()
