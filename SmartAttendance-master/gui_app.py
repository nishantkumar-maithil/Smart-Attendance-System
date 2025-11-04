import cv2
import threading
from datetime import datetime
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox, ttk
from face_utils import findEncodings, load_student_images
from database import (
    markAttendanceMySQL,
    fetch_attendance_by_date,
    delete_attendance_record,
    delete_all_attendance,
    getAllAttendance
)


class SmartAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("960x720")
        self.root.configure(bg="#eef2f3")

        # --- Initialize Variables ---
        self.images = []
        self.studentNames = []
        self.encodeListKnown = []
        self.path = ""
        self.running = False
        self.camera_thread = None
        self.stop_event = threading.Event()

        # --- UI Setup ---
        self.create_widgets()

    # --------------------- GUI Setup ---------------------
    def create_widgets(self):
        # Header
        Label(
            self.root,
            text="SMART ATTENDANCE SYSTEM",
            font=("Segoe UI", 24, "bold"),
            bg="#1A237E",
            fg="white",
            pady=12
        ).pack(fill=X)

        # Button Section
        frame = Frame(self.root, bg="#eef2f3", pady=20)
        frame.pack()

        button_config = {
            "font": ("Segoe UI", 12, "bold"),
            "fg": "white",
            "width": 25,
            "relief": GROOVE,
            "cursor": "hand2",
            "pady": 5
        }

        Button(frame, text="📂 Select Image Folder", command=self.load_images,
               bg="#5c6bc0", **button_config).grid(row=0, column=0, padx=10, pady=10)
        Button(frame, text="🧠 Encode Faces", command=self.encode_faces,
               bg="#3949ab", **button_config).grid(row=0, column=1, padx=10, pady=10)
        Button(frame, text="🎥 Start Attendance", command=self.start_attendance,
               bg="#1e88e5", **button_config).grid(row=1, column=0, padx=10, pady=10)
        Button(frame, text="🛑 Stop Attendance", command=self.stop_attendance,
               bg="#c62828", **button_config).grid(row=1, column=1, padx=10, pady=10)
        Button(frame, text="📅 View Today’s Attendance", command=self.show_today_attendance,
               bg="#43a047", **button_config).grid(row=2, column=0, padx=10, pady=10)
        Button(frame, text="📆 View By Date", command=self.show_by_date,
               bg="#00897b", **button_config).grid(row=2, column=1, padx=10, pady=10)
        Button(frame, text="📋 Show All Attendance", command=self.show_all_attendance,
               bg="#0288d1", **button_config).grid(row=3, column=0, padx=10, pady=10)
        Button(frame, text="🗑️ Delete Selected", command=self.delete_selected_attendance,
               bg="#f57c00", **button_config).grid(row=3, column=1, padx=10, pady=10)
        Button(frame, text="🧹 Delete All", command=self.delete_all_records,
               bg="#d32f2f", **button_config).grid(row=4, column=0, padx=10, pady=10)
        Button(frame, text="❌ Exit", command=self.root.quit,
               bg="#757575", **button_config).grid(row=4, column=1, padx=10, pady=10)

        # Status Label
        self.status_label = Label(self.root, text="Status: Ready", font=("Segoe UI", 12),
                                  bg="#eef2f3", fg="#212121")
        self.status_label.pack(pady=5)

        # Table Style
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("Segoe UI", 12, "bold"))
        style.configure("Treeview", font=("Segoe UI", 11), rowheight=28)

        # Attendance Table
        self.tree = ttk.Treeview(self.root, columns=("Name", "Date", "Time"), show='headings', height=12)
        for col, width in [("Name", 250), ("Date", 200), ("Time", 200)]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor=CENTER)
        self.tree.pack(fill=BOTH, padx=20, pady=10)

        self.status_label.config(text="Status: Ready — Click a button to manage attendance.")

    # --------------------- Load Images ---------------------
    def load_images(self):
        self.path = filedialog.askdirectory(title="Select Folder with Student Images")
        if not self.path:
            return
        self.images, self.studentNames = load_student_images(self.path)
        self.status_label.config(text=f"Loaded {len(self.images)} images.")
        messagebox.showinfo("Images Loaded", f"{len(self.images)} student images loaded successfully!")

    # --------------------- Encode Faces ---------------------
    def encode_faces(self):
        if not self.images:
            messagebox.showwarning("Warning", "Please load images first!")
            return
        self.status_label.config(text="Encoding faces, please wait...")
        self.root.update()
        self.encodeListKnown = findEncodings(self.images)
        self.status_label.config(text=f"Encoding completed — {len(self.encodeListKnown)} faces encoded.")
        messagebox.showinfo("Encoding", "Face encoding completed successfully!")

    # --------------------- Start Attendance ---------------------
    def start_attendance(self):
        if not self.encodeListKnown:
            messagebox.showwarning("Warning", "Please encode faces first!")
            return
        if self.running:
            messagebox.showinfo("Info", "Camera is already running.")
            return

        self.running = True
        self.stop_event.clear()
        self.status_label.config(text="Camera started — press Stop Attendance to end.")
        self.camera_thread = threading.Thread(target=self.run_camera, daemon=True)
        self.camera_thread.start()

    # --------------------- Stop Attendance ---------------------
    def stop_attendance(self):
        if not self.running:
            messagebox.showinfo("Info", "Camera is not running.")
            return
        self.stop_event.set()
        self.running = False
        self.status_label.config(text="Camera stopped successfully.")
        messagebox.showinfo("Stopped", "Camera stopped successfully!")

    # --------------------- Run Camera ---------------------
    def run_camera(self):
        import face_recognition
        cap = cv2.VideoCapture(0)
        THRESHOLD = 0.45
        marked_today = set()

        while not self.stop_event.is_set():
            success, img = cap.read()
            if not success:
                break

            imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                if len(faceDis) == 0:
                    continue

                matchIndex = np.argmin(faceDis)
                best_match_distance = faceDis[matchIndex]

                name = "UNKNOWN"
                if best_match_distance < THRESHOLD:
                    name = self.studentNames[matchIndex].upper()
                    if name not in marked_today:
                        try:
                            markAttendanceMySQL(name)
                            marked_today.add(name)
                            self.status_label.config(text=f"Marked: {name}")
                        except Exception as e:
                            print("Database Error:", e)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Smart Attendance System - Press 'q' to Stop", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False
        self.status_label.config(text="Camera stopped.")

    # --------------------- Show Today’s Attendance ---------------------
    def show_today_attendance(self):
        today = datetime.now().date()
        rows = fetch_attendance_by_date(today)
        self.display_records(rows, f"Today's Attendance ({today})")

    # --------------------- Show All Attendance ---------------------
    def show_all_attendance(self):
        rows = getAllAttendance()
        self.display_records(rows, "All Attendance Records")

    # --------------------- Show By Date ---------------------
    def show_by_date(self):
        top = Toplevel(self.root)
        top.title("Search Attendance by Date")
        top.geometry("300x150")
        top.resizable(False, False)
        Label(top, text="Enter Date (YYYY-MM-DD):", font=("Segoe UI", 12)).pack(pady=10)
        entry = Entry(top, font=("Segoe UI", 12))
        entry.pack(pady=5)

        def search():
            date_val = entry.get().strip()
            if not date_val:
                messagebox.showwarning("Warning", "Please enter a date!")
                return
            rows = fetch_attendance_by_date(date_val)
            self.display_records(rows, f"Attendance for {date_val}")
            top.destroy()

        Button(top, text="Search", command=search, font=("Segoe UI", 12, "bold"),
               bg="#3949ab", fg="white", width=12, cursor="hand2").pack(pady=10)

    # --------------------- Delete Selected ---------------------
    def delete_selected_attendance(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a record to delete!")
            return

        for item in selected:
            name, date, time = self.tree.item(item, "values")
            delete_attendance_record(name, date, time)
            self.tree.delete(item)

        messagebox.showinfo("Deleted", "Selected record(s) deleted successfully!")
        self.status_label.config(text="Selected record(s) deleted.")

    # --------------------- Delete All ---------------------
    def delete_all_records(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all records?"):
            delete_all_attendance()
            self.tree.delete(*self.tree.get_children())
            messagebox.showinfo("Deleted", "All attendance records deleted!")
            self.status_label.config(text="All records deleted.")

    # --------------------- Display Helper ---------------------
    def display_records(self, rows, msg):
        self.tree.delete(*self.tree.get_children())
        if not rows:
            messagebox.showinfo("No Records", "No attendance records found.")
        for r in rows:
            self.tree.insert("", END, values=r)
        self.status_label.config(text=f"Showing: {msg}")


# --------------------- Run App ---------------------
if __name__ == "__main__":
    root = Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()
