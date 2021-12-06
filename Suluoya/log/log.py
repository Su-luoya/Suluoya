import os
from time import asctime, localtime, time
from random import choice
import time
import pretty_errors
from termcolor import colored


def progress_bar(df):
    '''进度条，传入dataframe，返回可迭代对象'''
    scale = len(df)
    start = time.perf_counter()
    i=0
    for j in df.itertuples():
        i+=1
        print(f"\r{((i+1) / scale) * 100:^3.0f}%[{'*' * (i+1)}->{'.' * (scale - i-1)}]{time.perf_counter() - start:.2f}s",end = "")
        yield j

def makedir(path,name):
    try:
        os.mkdir(f'{path}\\{name}')
    except:
        pass

def hide():
    print(f'\033[0;30;40m',end='')

def show():
    print(colored('', 'white'), end='')


def sprint(content, color=False):
    '''Color Output
    '''
    color_list = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
    if not color:
        print(colored(content, color=choice(color_list)))
    else:
        print(colored(content, color=color))


class slog(object):
    def __init__(self, filename='Suluoya'):
        try:
            os.makedirs('./slog')
        except:
            pass
        self.filename = filename
        with open(f'slog\\{self.filename}.log', 'w', encoding='utf8') as f:
            f.write(asctime(localtime(time()))+'\n')
            f.write("(｡･∀･)ﾉﾞ嗨!\n\n")

    def log(self, content='Suluoya', mode=0):
        '''
        Logs a message
        0:\\n{content}\\n
        1:\\n{content}
        2:  {content }  
        '''
        mode_dict = {0: f'\n{content}\n',
                     1: f'\n{content}',
                     2: f'{content} '}
        with open(f'slog\\{self.filename}.log', 'a', encoding='utf8') as f:
            f.write(mode_dict[mode])


if __name__ == '__main__':
    # sprint('Suluoya')
    # slog = slog(filename='Suluoya')
    # slog.log('se', 2)
    makedir(r'C:\Users\19319\Desktop\sly\Data','data')
