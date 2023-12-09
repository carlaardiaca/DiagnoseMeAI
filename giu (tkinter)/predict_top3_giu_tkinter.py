import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tkinter import  ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from joblib import load

sys.path.insert(0, '../')
from clean_data import clean_text
from data_preprocessing import apply_tokens, apply_padding, one_hot_encode


# Carregar els models i els recursos necessaris
clf = load('../data/best_random_forest_model.joblib')
paraules_freq = load('../data/paraules_freq.joblib')
paraules_dicc = load('../data/paraules_dicc.joblib')
label_noms = load('../data/label_noms.joblib')

max_length = 27

fig, ax = plt.subplots(figsize=(9, 4))

def create_bar_chart(labels, probabilities, canvas):
    ax.clear()  # Clear the axes for the new plot
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, probabilities, color=plt.cm.Paired.colors, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')

    for i, v in enumerate(probabilities):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center', color='black')

    plt.tight_layout()
    canvas.draw()
    
def predir():
    ini_text = entrance_text.get("1.0", tk.END).strip()
    if not ini_text:
        messagebox.showerror("Error", "Please, enter some text to analyze.")
        return
    words = ini_text.split()
    if len(words) < 10:
        messagebox.showerror("Error", "Please enter more detailed symptoms (I need more than 10 words to make the prediction).")
        return

    processed_text = clean_text(ini_text)
    tokens = apply_tokens(processed_text, paraules_freq)
    tokens_padded = apply_padding([tokens], max_length)
    tokens_one_hot = one_hot_encode(tokens_padded, len(paraules_freq))
    data = pd.DataFrame(tokens_one_hot, columns=paraules_dicc)
    prediction = clf.predict_proba(data)[0]
    top_indices = np.argsort(prediction)[-3:][::-1]
    top_probabilities = prediction[top_indices]
    
    nom_num = {v: k for k, v in label_noms.items()}
    
    prediction_text = "Top 3 Predictions:\n"
    for i, idx in enumerate(top_indices):
        disease_name = nom_num.get(idx)  
        probability = prediction[idx]
        if i < len(top_indices) - 1:
            prediction_text += f"{disease_name}: {probability:.2f}\n"
        else:
            prediction_text += f"{disease_name}: {probability:.2f}"
    
    exit_label.config(text=prediction_text)     
    top_labels = [nom_num.get(int(idx)) for idx in top_indices]
    top_probabilities = prediction[top_indices]
    create_bar_chart(top_labels, top_probabilities, canvas)


def on_close():
    window.destroy()  # Destroy the main window
    plt.close('all')  # Close all Matplotlib figures
    sys.exit(0)   

def update_entrance_text_height(event=None):
    line_count = int(entrance_text.index('end-1c').split('.')[0])
    entrance_text.config(height=line_count)

def clear_all():
    entrance_text.delete('1.0', tk.END)
    exit_label.config(text="")  

    ax.clear()
    fig.canvas.draw_idle()  # Redraw the entire figure

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks
    canvas.draw()
    
window = tk.Tk() #instància de la classe Tk
window.title("Diagnosis of Disease")    
window.geometry('1700x1000') #amplada * altura
window.configure(background="white")
frame_entrance = ttk.Frame(window, style='TFrame')
window.protocol("WM_DELETE_WINDOW", on_close) 


# Estil/ Disseny


title_frame = ttk.Frame(window, style='TFrame')
title_frame.pack(fill='x', expand=False, padx=20, pady=5)

title_label = ttk.Label(title_frame, text="Please input your symptoms in the box below, and I will attempt to predict your disease:", font=('Courier', 14))
title_label.pack(side='top', pady=5)

frame_entrance = ttk.Frame(window)
frame_entrance.pack(fill='both', expand=True, padx=20, pady=10)

frame_exit = ttk.Frame(window)
frame_exit.pack(fill='both', expand=True, padx=20, pady=10)

entrance_text = tk.Text(frame_entrance, height=10, width=50, font=('Courier', 14), wrap='word', borderwidth=2, relief="groove")
entrance_text.pack(padx=10, pady=10, fill='both', expand=True)
entrance_text.bind("<KeyRelease>", update_entrance_text_height)

reset_frame = tk.Frame(window)
reset_frame.pack(fill='x', side='bottom', anchor='e', padx=10, pady=10)


# Using tk.Button for the predict button
predict_button = tk.Button(frame_entrance, text="Predict disease", font=('Courier', 14), bg='#00a8e8', fg='white', command=predir)
predict_button.pack(fill='x', expand=True)

exit_label = ttk.Label(frame_exit, text="", font=('Courier', 14), background='white', anchor='center', justify='center')
exit_label.pack(fill='both', expand=True)

chart_frame = tk.Frame(window)
chart_frame.pack(fill='both', expand=True)
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=chart_frame)  
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

canvas.get_tk_widget().pack(side=tk.TOP, fill='both', expand=True)

# Using tk.Button for the reset button
reset_button = tk.Button(reset_frame, text="Clear", font=('Courier', 14), bg='#00a8e8', fg='white', command=clear_all)
reset_button.pack(side='right')


window.mainloop() #refrescar la finestra