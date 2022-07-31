#working 

print("hello world")

new_pin = int(input("Enter a new pin: "))
while True:
    chances = 3
    pin = int(input("Enter your pin: "))
    if chances > 0:
        if new_pin == pin:
            print("correct")
            break
            
        else:
            print("incorrect")
            chances -=1
            print(chances)
            if chances == 0:
                break
    