print("-----《Python编程从入门到实践》P149 练习9-4 就餐人数-----")


class Restaurant:
    def __init__(self, name, c_type):
        self.restaurant_name = name
        self.cuisine_type = c_type
        self.number_served = 0

    def describe_restaurant(self):
        print("restaurant neme is", self.restaurant_name)
        print("cuisine type is", self.cuisine_type)

    def open_restaurant(self):
        print("restaurant is opening")

    def set_number_served(self, num):
        self.number_served = num


restaurant = Restaurant("锦龙饭店", "猪脚饭")
restaurant.describe_restaurant()
restaurant.open_restaurant()
print("就餐人数：", restaurant.number_served)
restaurant.set_number_served(10)
print("就餐人数：", restaurant.number_served)


print("-----《Python编程从入门到实践》P149 练习9-5 尝试登录次数-----")


class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
        self.login_attempts = 0

    def describe_user(self):
        print(f"My first name is {self.first_name}, my last name is {self.last_name}.")

    def greet_user(self):
        print("Welcome!")

    def increment_login_attempts(self):
        self.login_attempts += 1

    def reset_login_attempts(self):
        self.login_attempts = 0


Luoliming = User("Liming", "Luo")
Luoliming.describe_user()
Luoliming.greet_user()
for i in range(5):
    Luoliming.increment_login_attempts()
print("尝试登录次数：", Luoliming.login_attempts)
Luoliming.reset_login_attempts()
print("登录次数复位：", Luoliming.login_attempts)


print("-----《Python编程从入门到实践》P155 练习9-6 冰激凌小店-----")


class IceCreamStand(Restaurant):
    def __init__(self, name, c_tpye):
        super().__init__(name, c_tpye)
        self.flavors = ["草莓味", "抹茶味", "原味"]

    def show_flavors(self):
        print(f"{self.restaurant_name}拥有{self.cuisine_type}口味: {self.flavors}")


IceCreamShop = IceCreamStand("梁工的小店", "冰激凌")
IceCreamShop.describe_restaurant()
IceCreamShop.show_flavors()


print("-----《Python编程从入门到实践》P155 练习9-7 管理员-----")


class Admin(User):
    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
        self.privileges = Privileges()


# admin = Admin("Liming", "Luo")
# admin.show_privileges()


print("-----《Python编程从入门到实践》P155 练习9-8 权限-----")


class Privileges:
    def __init__(self):
        self.privileges = ["can add post", "can delete post", "can ban user"]

    def show_privileges(self):
        print("管理员权限：", self.privileges)


admin = Admin("Liming", "Luo")
admin.privileges.show_privileges()