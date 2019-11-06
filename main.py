import os
import sys
import time
import tkinter
from tkinter import Scale
from tkinter import filedialog
from tkinter import messagebox
from tkinter import PhotoImage
from ttkthemes import themed_tk as tk
import tkinter.ttk as ttk
import subprocess
import queue
import threading
import pandas
import numpy as np
import Cosine_Similarity
import re
import datetime
from pathlib import Path
import gc
def getDataFrameWords(df):
    return df.lines.str.strip().str.split('[\W_]+')

def tfidf(filePathDictionary, fileTree, first,chunksize):
    fpd = pandas.concat(filePathDictionary)
    fpd.to_csv(filepath + "\\pathDictionary.csv",
    header=first,
    mode='a',#append data to csv
    chunksize=chunksize)
    df = pandas.concat(fileTree)
    df['words'] =  getDataFrameWords(df)## partiendo las palabras a pedazos
    rows = list()
    for row in df[['path', 'words']].iterrows():
        r = row[1]
        for word in r.words:
            if len(word) > 0 and word.isalpha():
                rows.append((r.path, word.lower()))

    words = pandas.DataFrame(rows,columns=['path','word'])
    counts = words.groupby('path')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})

    word_sum = counts.groupby(level=0)\
        .sum()\
        .rename(columns={'n_w': 'n_d'})

    tf = counts.join(word_sum)

    tf['tf'] = tf.n_w/tf.n_d
    tf = tf.sort_values('path')
    tf.to_csv(filepath +  "\\tf.csv",
    header=first,
    mode='a',#append data to csv
    chunksize=chunksize)#size of data to append for each loop

    idfTemp= words.groupby('word')\
    .path\
    .nunique()\
    .to_frame()\
    .rename(columns={'path':'i_d'})\
    .sort_values('i_d')
            
    return idfTemp

def getAmountOfFiles(mainWindow):
    return sum([len(files) for r, d, files in os.walk(mainWindow.searchFolder)])
def writer(mainWindow):  ##creates index
    fileTree = list()
    filePathDictionary = list()
    first = True
    #size of chunks of data to write to the csv
    contador = 1
    chunksize = 25000
    fileamount = getAmountOfFiles(mainWindow)
    mainWindow.splash.pb["maximum"] = fileamount
    mainWindow.splash.fileText['text'] = str(contador) + " file(s) processed out of " + str(fileamount)
    mainWindow.splash.update()
    idf = pandas.DataFrame()
    startTime = time.time()
    
    for file in Path(mainWindow.searchFolder).rglob('*.*'):
        if contador % 2500 == 0:
            mainWindow.splash.pb["value"] = contador
            mainWindow.splash.fileText['text'] = str(contador) + " file(s) processed out of " + str(fileamount)
            mainWindow.splash.update()
        if contador % chunksize == 0 or contador == fileamount:

            idfTemp = tfidf(filePathDictionary, fileTree, first,chunksize)
            
            if first:
                idf = idfTemp.copy()
            else:
                idf = pandas.concat([idf, idfTemp],sort = False)
                
            if not first or contador == fileamount:
                
                idf['i_d'] = idf.groupby('word')['i_d'].transform('sum')
                idf = idf[~idf.index.duplicated(keep='first')]

            fileTree.clear()
            filePathDictionary.clear()
            
            first = False
            
        with open(file.resolve(), encoding='utf-8') as f:
            fileTree.append(pandas.DataFrame({'path': contador - 1, 'lines': f.readlines()}))
            filePathDictionary.append(pandas.DataFrame({'path': file.resolve(), 'index': contador - 1},index=[contador - 1]))
        contador += 1

    idf['idf'] = np.log(contador - 1/idf.i_d.values)
    idf.to_csv(filepath +  "\\idf.csv")
    first = True
    for chunk in pandas.read_csv(filepath +  "\\tf.csv", chunksize=chunksize):
        tf_idf = pandas.merge(chunk, idf, on='word', sort=False)
        tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf
        compact = tf_idf.drop(tf_idf.columns[[2,3,4,5,6]], 1).sort_values('path')
        compact.to_csv(filepath +  "\\tf-idf.csv",
        header=first,
        mode='a',#append data to csv
        chunksize=chunksize)#size of data to append for each loop
        first = False
    endTime = time.time()
    os.remove(filepath + "\\tf.csv")
    timeTaken = endTime - startTime
    tkinter.messagebox.showinfo(title = "Success",\
                                   message = "Created index for " + str(fileamount) + " document(s)\nin "+ "{0:.2f}".format(timeTaken) + "s")

def vectorizeQuery(query, words):

    rows = list()
    for word in words:
        if len(word) > 0 and word.isalpha():
            rows.append((1, word.lower()))
    wdf = pandas.DataFrame(rows, columns=['path','word'])
    counts = wdf.groupby('path')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})

    word_sum = counts.groupby(level=0)\
        .sum()\
        .rename(columns={'n_w': 'n_d'})

    tf = counts.join(word_sum)

    tf['tf'] = tf.n_w/tf.n_d
    idf = pandas.read_csv(filepath + "\\idf.csv")

    tf_idf = pandas.merge(tf, idf, on='word', sort=False,how='inner')
    tf_idf = tf_idf.fillna(0)
    tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf
    return tf_idf.drop(tf_idf.columns[[1,2,3,4,5]], 1).sort_values(by=['word'])

class FolderSelect:
    def __init__(self, father):
        self.searchPath = ""
        if father:
            if father.newIndex:
                self.searchPath = tkinter.filedialog.askdirectory(parent=father,initialdir="/",title='Please select a directory')
                self.searchPath = self.searchPath.replace("/","\\")

class EntryWithPlaceholder(ttk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey'):
        super().__init__(master, width = "50")

        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = 'black'

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        self.put_placeholder()

    def put_placeholder(self):
        self.insert(0, self.placeholder)
        self.configure(foreground= self.placeholder_color)
        self.foreground = self.placeholder_color

    def foc_in(self, *args):
        if self.foreground == self.placeholder_color:
            self.delete('0', 'end')
            self.configure(foreground= self.default_fg_color)
            self.foreground = self.default_fg_color

    def foc_out(self, *args):
        if not self.get():
            self.put_placeholder()

class ThreadedTask:
    def __init__(self,parent, queue):
        self.active = True
        self.updating = True
        parent.updating = True
        self.queue = queue
        self.parent = parent
        self.parent.task = self
        self.thread1 = threading.Thread(target=self.workerThread1)
        self.thread2 = threading.Thread(target=self.updateProgressBar)
        self.parent.splash = Splash(self.parent, "Searching...")
        self.contador = 0
        self.fileamount = sum(1 for line in open(filepath + "\\tf-idf.csv"))
        self.parent.splash.pb["maximum"] = self.fileamount
        self.parent.splash.fileText['text'] = str(self.contador) + " chunk(s) searched out of " + str(self.fileamount)
        self.parent.splash.update()
        self.thread1.start()
        self.thread2.start()
        self.startTime = time.time()
        self.endTime = time.time()

    def updateProgressBar(self):
        if(self.updating):
            self.parent.splash.pb["value"] = self.contador
            self.parent.splash.fileText['text'] = str(self.contador) + " chunk(s) processed out of " + str(self.fileamount)
            self.parent.splash.update()
            self.parent.after(750, self.updateProgressBar)

    def workerThread1(self):
        """
        This is where we handle the asynchronous I/O. For example, it may be
        a 'select(  )'. One important thing to remember is that the thread has
        to yield control pretty regularly, by select or otherwise.
        """
        self.contador = 0


        chunksize = 1000000
        if(os.path.isfile(filepath + "\\queryResult.csv")):
            os.remove(filepath + "\\queryResult.csv")
        querystr = self.parent.my_entry.lower().strip()
        regex = re.compile('[^a-zA-Z ]')
        querystr = regex.sub('', querystr)
        words = querystr.split(' ')
        vectorized = vectorizeQuery(querystr,words)
        words = vectorized[vectorized.columns[0]].values
        wordDf = pandas.DataFrame(words, columns=['word'])
        size = wordDf.shape[0]
        queryVector = vectorized[vectorized.columns[-1]].values
        path = ''
        errorPercentage = int(self.parent.slider.get()) / 100
        first = True
        porcentaje = 0
        if(queryVector.all()):
            for chunk in pandas.read_csv(filepath + "\\tf-idf.csv", chunksize= chunksize,  index_col = 0):
                answer = list()

                sample = chunk.sample(int(chunk.shape[0] * errorPercentage))
                sample = pandas.merge(sample, wordDf, on='word', sort=False,how='inner')
                if(sample.shape[0]):
                    porcentaje =int(chunk.shape[0]/ sample.shape[0])
                sample = sample.groupby(['path'])

                for pathNum, tfidf in sample:
                    self.contador += tfidf.shape[0] * porcentaje
                    if(tfidf.shape[0] == size):
                        column = tfidf[tfidf.columns[-1]].values
                        number = Cosine_Similarity.cosine_similarity(column,queryVector).item()
                        answer.append((pathNum, number))

                answerDataFrame = pandas.DataFrame(answer, columns=["index","similarity"]).dropna()
                answerDataFrame = answerDataFrame[~answerDataFrame.similarity.duplicated(keep='first')]
                answerDataFrame.to_csv(filepath + "\\queryResult.csv",
                header=first,
                mode='a',#append data to csv
                chunksize=chunksize, index=False)
                first = False

        self.updating = False
        self.parent.updating = False
        self.endTime = time.time()
        self.queue.put("done")
        self.parent.splash.destroy()
        self.parent.process_queue()


class Splash(tkinter.Toplevel):
    def __init__(self, parent, message):
        tkinter.Toplevel.__init__(self, parent)
        self.title("Please wait")
        self.Text = tkinter.Label(self, text = message, font= ("Helvetica", 16))
        self.Text.place(x=100,y=75,anchor="center")
        self.pb = ttk.Progressbar(self, orient="horizontal", length=150, mode="determinate")
        self.pb.place(x=100,y=100,anchor="center")
        self.fileText = tkinter.Label(self, text = "", font= ("Helvetica", 6))
        self.fileText.place(x=100,y=125,anchor="center")
        self.geometry("200x200")
        ## required to make window show before the program gets to the mainloop

        self.update()

class MainGUI(tk.ThemedTk):
    """ The GUI """

    def search_button(self):
        if(self.task):
            self.task.active = False
        self.results.clear() #cleared because of next searches
        self.my_entry = self.ent.get()
        if not self.my_entry:
            tkinter.messagebox.showwarning(title = "No keyword to Search" \
                                           ,message = "Enter a Keyword")
            return None
        #Searches for the query in new thread so GUI doesn't become irresponsive
        ThreadedTask(self,self.queue)


    def on_frame_configure(self, event):
        self.cvs.configure(scrollregion=self.cvs.bbox("all"))

    def __init__(self):
        super().__init__()
        self.newIndex = not fileTreeExists
        self.menubar = tkinter.Menu(self)
        self.task = None
        self.searchFolder = None
        self.splash = None
        filemenu = tkinter.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Create Index", command=self.indexCreation)
        self.menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=self.menubar)
        self.queue = queue.Queue()
        self.updating = False


        self.folder = FolderSelect(self)
        self.searchFolder = self.folder.searchPath

        if(not self.searchFolder and self.newIndex):
            self.destroy()
            return

        if self.newIndex:   ##checks if the file tree has been created
            self.writeIndex()

        self.results = []
        self.logoImage = PhotoImage(file=filepath + "\\yoogling_logo.png")
        self.style = ttk.Style()
        self.style.theme_use('arc')
        self.logo = ttk.Label(self, image=self.logoImage)
        self.configure(background="#f5f6f7")
        self.ent = EntryWithPlaceholder(master = self, placeholder = "Query")
        self.btn = ttk.Button(self, text="Search", command=self.search_button)
        self.cvs = tkinter.Canvas(self, borderwidth=0, background="#ffffff")
        self.sliderFrame = tkinter.Frame(self, background="#f5f6f7")
        self.frame = tkinter.Frame(self.cvs, background="#ffffff")


        self.sliderText = ttk.Label(self.sliderFrame, text = "Precision percentage:")
        self.sliderValue = ttk.Label(self.sliderFrame, text = "100")
        self.slider = ttk.Scale(self.sliderFrame, from_=0, to=100, orient='horizontal',  length=130,command=self.updateValue)
        self.slider.set(100)
        self.vsb = ttk.Scrollbar(self.cvs, orient="vertical", command=self.cvs.yview)
        self.cvs.configure(yscrollcommand=self.vsb.set)
        self.logo.pack(padx=10,pady=(25,25))
        self.ent.pack(padx=10,pady=5)
        self.sliderFrame.pack(fill='x', padx = 20, pady = (0,10))
        self.sliderText.pack(side = 'left', pady = (15,0), padx=(10,10))
        self.slider.pack(side = 'left',pady = (15,0))
        self.sliderValue.pack(side = 'right', pady = (15,0), padx=(0,30))


        self.btn.pack(padx=10,pady=5)
        self.vsb.pack(side="right", fill="y", pady=2)
        self.cvs.pack(fill="both", expand=True,padx=5,pady=5)
        self.cvs.create_window((4,4), window=self.frame, anchor="nw",
                                  tags="self.frame")
        self.lst = tkinter.Listbox(self.frame, selectmode="SINGLE", height = len(self.results), width = "100", borderwidth=0, highlightthickness=0, selectbackground = "white", selectforeground = "black")
        self.frame.bind("<Configure>", self.on_frame_configure)
        self.lst.pack(fill="both", expand=True)
        self.lst.bind("<Double-Button-1>", self.open_folder)
        self.geometry("400x500")
        self.resizable(False,False)

        self.mainloop()

    def indexCreation(self):
        self.newIndex = True
        self.searchFolder = None

        self.folder = FolderSelect(self)
        self.searchFolder = self.folder.searchPath
        if not self.searchFolder:
            self.destroy()
            return

        if(os.path.isfile(filepath + "\\queryResult.csv")):
            os.remove(filepath + "\\queryResult.csv")
        if(os.path.isfile(filepath + "\\tf-idf.csv")):
            os.remove(filepath + "\\tf-idf.csv")
        if(os.path.isfile(filepath + "\\pathDictionary.csv")):
            os.remove(filepath + "\\pathDictionary.csv")
        if(os.path.isfile(filepath + "\\tf.csv")):
            os.remove(filepath + "\\tf.csv")
        if(os.path.isfile(filepath + "\\idf.csv")):
            os.remove(filepath + "\\idf.csv")
        self.writeIndex()
        self.newIndex = False

    def writeIndex(self):

        self.withdraw()
        self.splash = Splash(self, "Initializing index")
        writer(self)
        self.splash.destroy()
        self.deiconify()

    def updateValue(self, event):
        self.sliderValue.config(text = str(int(self.slider.get())))

    def process_queue(self):
        if self.updating or (not self.queue.empty()):

            if(not self.queue.empty()):
                while(not self.queue.empty()):
                    self.queue.get(0)
                self.results = []
                if(os.path.isfile(filepath + "\\queryResult.csv")):
                    result = pandas.read_csv(filepath + "\\queryResult.csv",  index_col = 0)
                    rutas = pandas.read_csv(filepath + "\\pathDictionary.csv",  index_col = 0)
                    final = result.join(rutas)
                    final = final.drop(final.columns[[2]], axis=1).sort_values('similarity', ascending=False)
                    amount = final.shape[0]
                    self.results = final[final.columns[-1]].values.tolist()

            self.display_results()

        elif(self.task.active):
            self.display_results()

    def display_results(self):
        
        if not self.results and not self.task.updating and self.task.active:
            time =  self.task.endTime - self.task.startTime
            self.task.active = False
            tkinter.messagebox.showwarning(title = "Could not find",\
                                           message = "There is no result matching with the keyword\nthe search lasted " + "{0:.2f}".format(time) + "s")
            return None
        if self.task.active and self.results:
            time =  self.task.endTime - self.task.startTime
            self.lst.delete('0', 'end')
            for index, item in enumerate(self.results, start=1):
                self.lst.insert(index, item.split("\\")[-1])
            self.task.active = False
            amount = len(self.results)
            tkinter.messagebox.showinfo(title = "Success",\
                                           message = "Matched " + str(amount) + " document(s)\nin "+ "{0:.2f}".format(time) + "s")

    def open_folder(self,event): #this opens folder with selected item using shell commands
        selected_path = self.results[event.widget.curselection()[0]]
        subprocess.Popen(r'explorer /select,' + selected_path)

if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    fileTreeExists = os.path.isfile(filepath + "\\tf-idf.csv")
    GUI = MainGUI()
