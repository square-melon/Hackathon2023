import tkinter
import tkinter.messagebox
import customtkinter
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageEnhance, ImageFilter
import os
import openai
import re
import matplotlib.pyplot as plt
from posixpath import join
from bing_image_downloader import downloader
import requests
from requests_html import HTMLSession
import re
import rich
import rich.table
import urllib.parse
from bs4 import BeautifulSoup
import ptt

openai.api_key = "***" #use own key
openai.organization = "org-dKb6Se0pRzf3hccYKKg8awJG"

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
font_size = 50
font1 = ImageFont.truetype('jf-openhuninn-2.0.ttf', font_size)  #可愛字
font2 = ImageFont.truetype('微軟正黑體-1.ttf', font_size)  
font3 = ImageFont.truetype('TaipeiSans.ttf', font_size)  #黑體

url = 'https://www.ptt.cc/bbs/Gossiping/index.html'
session = HTMLSession()
session.cookies.set('over18', '1')  # 向網站回答滿 18 歲了 !

# 發送 HTTP GET 請求並獲取網頁內容
response = session.get(url)
print(response.text)
controls = response.html.find('.action-bar a.btn.wide')



class App(customtkinter.CTk):
    
    def __init__(self):
        super().__init__()

        # configure window
        self.title("卡個懶人包")
        self.geometry(f"{1100}x{580}")
        self.new_string = "test"
        self.file_path = ""
        self.pttnews = ["news"]
        self.pttdick = {}
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.ptt_frame = customtkinter.CTkFrame(self.sidebar_frame, width=50, corner_radius=10, height=50)
        self.ptt_frame.grid(row=4, column=0, rowspan=2, sticky="nsew",pady=(10, 30),padx=(10,10))
        #self.ptt_frame.grid_rowconfigure(4, weight=1)
        self.appearance_mode_label2 = customtkinter.CTkLabel(self.ptt_frame, text="PTT Search", anchor="w")
        self.appearance_mode_label2.grid(row=0, column=0, padx=20, pady=(10, 0))

        self.ptt_button_1 = customtkinter.CTkButton(self.ptt_frame,text="Find Popular Post", command=self.find_ptt)
        self.ptt_button_1.grid(row=1, column=0, padx=20, pady=10)
        
        self.pttbox = customtkinter.CTkComboBox(self.ptt_frame,values=self.pttnews)
        self.pttbox.grid(row=2, column=0, padx=20, pady=10)

        self.ptt_button_2 = customtkinter.CTkButton(self.ptt_frame, text="Generate Popular Post", command=self.generate_ptt)
        self.ptt_button_2.grid(row=3, column=0, padx=20, pady=10)


        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="卡個懶人包", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame,text="Upload Image", command=self.upload_image)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Generate Image", command=self.generate_image)
        self.sidebar_button_3.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Download Image", command=self.download_image)
        self.sidebar_button_2.grid(row=3, column=0, padx=20, pady=10)
       
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        #self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        #self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="input article")
        self.entry.grid(row=3, column=1, padx=(20, 0), pady=(0, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),text="generate simplify article" ,command = self.update_article)
        self.main_button_1.grid(row=3, column=2, padx=(20, 20), pady=(0, 20), sticky="nsew")

        # create textbox
        self.img_label = tk.Label(self,width=500)
        self.img_label.grid(row=0, column=1, padx=(20, 0), pady=(20, 0))
        #self.textbox = customtkinter.CTkTextbox(self, width=250)
        #self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        
        # create tabview
        self.frame = customtkinter.CTkFrame(self, width=300)
        self.frame.grid(row=0, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        #self.tabview.add("Font Type")
        #self.tabview.tab("Font Type").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs

        self.combobox_1 = customtkinter.CTkComboBox(master=self.frame,
                                                        values=["黑體", "粉圓", "微軟正黑體"])
        self.combobox_1.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.combobox_2 = customtkinter.CTkComboBox(master=self.frame,
                                                    values=["12pt", "14pt", "16pt", "18pt", "20pt", "22pt", "24pt", "26pt", "28pt", "30pt"])
        self.combobox_2.grid(row=1, column=0, padx=20, pady=(10, 10))

        self.combobox_3 = customtkinter.CTkComboBox(master=self.frame,
                                                    values=["top", "middle", "bottom"])
        self.combobox_3.grid(row=2, column=0, padx=20, pady=(10, 10))

        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.frame,text="blurry background")
        self.checkbox_1.grid(row=3, column=0, padx=20, pady=(10, 10))

        self.string_input_button = customtkinter.CTkButton(master=self.frame, text="Add Text",
                                                           command=self.add_text)
        self.string_input_button.grid(row=4, column=0, padx=20, pady=(10, 10))
        
        '''
        # create radiobutton frame
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="CTkRadioButton Group:")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=0)
        self.radio_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=1)
        self.radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=2)
        self.radio_button_3.grid(row=3, column=2, pady=10, padx=20, sticky="n")
        '''

        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        '''
        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        self.seg_button_1 = customtkinter.CTkSegmentedButton(self.slider_progressbar_frame)
        self.seg_button_1.grid(row=0, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=1, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_2 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_2.grid(row=2, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.slider_1 = customtkinter.CTkSlider(self.slider_progressbar_frame, from_=0, to=1, number_of_steps=4)
        self.slider_1.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.slider_2 = customtkinter.CTkSlider(self.slider_progressbar_frame, orientation="vertical")
        self.slider_2.grid(row=0, column=1, rowspan=5, padx=(10, 10), pady=(10, 10), sticky="ns")
        self.progressbar_3 = customtkinter.CTkProgressBar(self.slider_progressbar_frame, orientation="vertical")
        self.progressbar_3.grid(row=0, column=2, rowspan=5, padx=(10, 20), pady=(10, 10), sticky="ns")
        '''
        self.textbox2 = customtkinter.CTkTextbox(self, width=125)
        self.textbox2.grid(row=1, column=2, padx=(20, 20), pady=(20, 0), sticky="nsew")
        '''
        # create scrollable frame
        self.scrollable_frame = customtkinter.CTkScrollableFrame(self, label_text="CTkScrollableFrame")
        self.scrollable_frame.grid(row=1, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame_switches = []
        for i in range(100):
            switch = customtkinter.CTkSwitch(master=self.scrollable_frame, text=f"CTkSwitch {i}")
            switch.grid(row=i, column=0, padx=10, pady=(0, 20))
            self.scrollable_frame_switches.append(switch)
        
        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_2.grid(row=2, column=0, pady=(20, 0), padx=20, sticky="n")
        self.checkbox_3 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_3.grid(row=3, column=0, pady=20, padx=20, sticky="n")
        '''
        # set default values
        #self.sidebar_button_3.configure(state="disabled", text="Disabled CTkButton")
        #self.checkbox_3.configure(state="disabled")
        #self.checkbox_1.select()
        #self.scrollable_frame_switches[0].select()
        #self.scrollable_frame_switches[4].select()
        #self.radio_button_3.configure(state="disabled")
        self.appearance_mode_optionemenu.set("Light")
        self.scaling_optionemenu.set("100%")
        #self.optionmenu_1.set("CTkOptionmenu")
        #self.combobox_1.set("CTkComboBox")
        '''
        self.slider_1.configure(command=self.progressbar_2.set)
        self.slider_2.configure(command=self.progressbar_3.set)
        self.progressbar_1.configure(mode="indeterminnate")
        self.progressbar_1.start()
        '''
        self.textbox.insert("0.0", "Original Article\n\n" )
        self.textbox2.insert("0.0", "Simplify Article\n\n" )
        #self.seg_button_1.configure(values=["CTkSegmentedButton", "Value 2", "Value 3"])
        #self.seg_button_1.set("Value 2")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")
    
    def download_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            img.save(file_path)

    def update_image(self):
        global img, img_label
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk
        print("do update")
    
    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            global img, img_label
            img = Image.open(self.file_path)
            img.thumbnail((500, 500))  # Resize the image to fit the GUI
            img_tk = ImageTk.PhotoImage(img)
            self.img_label.config(image=img_tk)
            self.img_label.image = img_tk
    
    def update_article(self):
        text_new = self.entry.get()
        self.textbox.delete("0.0","end")
        self.textbox.insert("0.0", "Original Article\n\n"+  text_new)
        simplify_text = self.gen_text(text_new)
        self.new_string = simplify_text
        self.textbox2.delete("0.0","end")
        self.textbox2.insert("0.0", "Simplify Article\n\n"+  simplify_text)
        

    def gen_text(self,user_input):
        message_history = []

        def chat(inp, role="user"):
            message_history.append({"role": role, "content": f"{inp}"})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message_history
            )
            reply_content = completion.choices[0].message.content
            message_history.append({"role": "assistant", "content": f"{reply_content}"})
            return reply_content

        message_history.append({"role": "system", "content": "你是個機器人，負責將新聞統整成三句話以內"})

        return chat(user_input)

    def add_text(self):
        if self.file_path:
            global img, img_label
            img = Image.open(self.file_path)
            img.thumbnail((500, 500))  # Resize the image to fit the GUI
            #img_tk = ImageTk.PhotoImage(img)
            #self.img_label.config(image=img_tk)
            #self.img_label.image = img_tk

        if self.checkbox_1.get():
            img = img.filter(ImageFilter.BLUR)

        #global img, img_label
        draw = ImageDraw.Draw(img)

        if self.combobox_1.get() == "粉圓":
            font_type = 'jf-openhuninn-2.0.ttf'
        elif self.combobox_1.get() == "微軟正黑體":
            font_type = '微軟正黑體-1.ttf'
        elif self.combobox_1.get() == "黑體":
            font_type = 'TaipeiSans.ttf'

        if self.combobox_2.get() == "12pt":
            font_size = 12
        elif self.combobox_2.get() == "14pt":
            font_size = 14
        elif self.combobox_2.get() == "16pt":
            font_size = 16
        elif self.combobox_2.get() == "18pt":
            font_size = 18
        elif self.combobox_2.get() == "20pt":
            font_size = 20
        elif self.combobox_2.get() == "22pt":
            font_size = 22
        elif self.combobox_2.get() == "24pt":
            font_size = 24
        elif self.combobox_2.get() == "26pt":
            font_size = 26
        elif self.combobox_2.get() == "28pt":
            font_size = 28
        elif self.combobox_2.get() == "30pt":
            font_size = 30

        
        
        #print(selected_font.get())    

        font = ImageFont.truetype(font_type, font_size) 
        #print(font_type,font_size,self.new_string)
        
        text_x, text_y = 10, 10
        text_color = "white"
        outline_colors = ["black"]  # 可以添加更多顏色
        thickness = 2.5/30*font_size
        new_string2 = re.split(r'[，。!?]', self.new_string)

        temp = img.size[1]-(len(new_string2)+1)*font_size
        if self.combobox_3.get() == "top":
            text_y = 10
        elif self.combobox_3.get() == "middle":
            text_y =( 10 + temp)/2
        elif self.combobox_3.get() == "bottom":
            text_y = temp
        

        for i, color in enumerate(outline_colors):
            outline_x = text_x - thickness + i
            outline_y = text_y - thickness + i
            outline_x2 = text_x + thickness + i
            outline_y2 = text_y + thickness + i
            for j, string in enumerate(new_string2):
                draw.text((outline_x, outline_y+(j+1)*font_size), string, fill=color, font = font)
                draw.text((outline_x2, outline_y2+(j+1)*font_size), string, fill=color, font = font)
        for j, string in enumerate(new_string2):
            draw.text((text_x, text_y+(j+1)*font_size), string, fill=text_color, font = font)
        #draw.text((10, 10), text_entry.get(), fill="black", font = font3)
        #self.update_image
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk

    def generate_image(self):
        message_history = []

        def chat(inp, role="user"):
            message_history.append({"role": role, "content": f"{inp}"})
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=message_history
            )
            reply_content = completion.choices[0].message.content
            message_history.append({"role": "assistant", "content": f"{reply_content}"})
            return reply_content
        message_history.append({"role": "system", "content": "You're a chinese who recognizes proper nouns in the chinese sentence and selected no more than 5 of them, and return the list of these words in python."})

        
        user_input = self.new_string     # News title.
        response = chat(user_input)
        print(response)
        pos1 = response.find("[")
        pos2 = response.find("]")
        if pos1==-1 or pos2==-1:
            string = user_input
    
        if pos1 == 0 and pos2 == 1:
            string = user_input
        
        delimiters = [",", '"', "'"]
        string = response[pos1+1: pos2]
        for delimiter in delimiters:
            string = "".join(string.split(delimiter))
        print(string)
        query_string = string
        downloader.download(query_string, limit=1,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)

        files = os.listdir('dataset/'+string)
        if files != []:
            self.file_path = 'dataset/'+string+'/'+files[0]
            if self.file_path:
                global img, img_label
                img = Image.open(self.file_path).convert("RGB")
                img.thumbnail((500, 500))  # Resize the image to fit the GUI
                img_tk = ImageTk.PhotoImage(img)
                self.img_label.config(image=img_tk)
                self.img_label.image = img_tk

    def find_ptt(self):
        self.pttdick=ptt.get_titleNlink()
        self.pttnews=list(self.pttdick.keys())
        self.pttbox.destroy()
        self.pttbox = customtkinter.CTkComboBox(self.ptt_frame,values=self.pttnews)
        self.pttbox.grid(row=2, column=0, padx=20, pady=10)
        #print(self.pttnews)


    def generate_ptt(self):
        answer = self.pttbox.get()
        o_article = self.pttdick[answer]
        o_article,temp = ptt.getcontent_withURL(o_article)
        self.textbox.delete("0.0","end")
        self.textbox.insert("0.0", "Original Article\n\n"+  o_article)

        simplify_text = self.gen_text(o_article)
        self.new_string = simplify_text
        self.textbox2.delete("0.0","end")
        self.textbox2.insert("0.0", "Simplify Article\n\n"+  simplify_text)



if __name__ == "__main__":
    app = App()
    app.mainloop()