import wandb
import time
import random

username = "nicolasfan"
ticker = 'My super panel'

wandb.init(project="cleanrl", name=ticker, id=wandb.util.generate_id())

stock_prices = [random.randint(0, 10) for _ in range(100)]
for idx, stock_price in enumerate(stock_prices):
    wandb.log({'epoch': idx, '666666': stock_price})

last_lenght = len(stock_prices)
stock_prices2 = [random.randint(0, 10) for _ in range(100)]
for idx, stock_price in enumerate(stock_prices2):
    wandb.log({'epoch': idx+last_lenght, '666666': stock_price})