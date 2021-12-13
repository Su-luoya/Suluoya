import pyperclip as pc
from random import choice

class Latex(object):

    def __init__(self):
        pass

    def get_clipboard(self):
        '''
        获取剪贴板数据
        '''
        data = pc.paste()
        return data

    def threeline_table(self):
        '''
        根据剪切板自动生成三线表latex语法
        '''
        data = self.get_clipboard()
        columns = len(data.split('\n')[0].split('\t'))
        length = 72/columns-6
        part1 = '\\setlength{\\tabcolsep}{'+str(int(length))+'mm}{\n\\begin{table}[H]\n  \\centering\n  \\caption{三线表}\n  \\resizebox{\\textwidth}{!}{\n    \\begin{tabular}'
        part2 = '{'+'c'*columns+'}'
        part3 = '\n      \\toprule[1.5pt]\n'
        data = data.replace('\t', ' & ').replace('\r\n', '\\\\\n').rstrip()
        data = data.replace('\\\\\n', '\\\\ \midrule[0.75pt]\n', 1)
        data += ' \\bottomrule[1.5pt]'
        data = data.replace('\n', '\n      ').lstrip()
        part4 = '\n    \\end{tabular}\n  }\n\\end{table}}'
        data = part1+part2+part3+'      '+data+part4
        result = data.strip()
        # print(result)
        pc.copy(result)

    def figure(self):
        '''
        根据剪切板路径自动插入图片latex语法
        '''
        part1 = '\\begin{figure}[H]\n  \centering\n  \includegraphics[scale=0.8]'
        path = '{'+self.get_clipboard().replace('\\', '/')+'}\n'
        part2 = '  \caption{'
        part_name = self.get_clipboard().split('\\')[-1].split('.')[0]
        part3 = '}\n\end{figure}'
        result = part1+path+part2+part_name+part3
        # print(result)
        pc.copy(result)


if __name__ == '__main__':

    latex = Latex()


    def work():
        while 1:
            f = input('Please choose a function, figure or threeline table? (f or t):')
            if f == 'f':
                latex.figure()
                break
            elif f == 't':
                latex.threeline_table()
                break
            else:
                print(
                    'Please choose from "f" for figure or "t" for threeline table!', color='red')
        print('')
        print('Already written to the clipboard!')


    while 1:
        work()
        print('ヾ(•ω•`)o'*5)
        print('================================================\n')
