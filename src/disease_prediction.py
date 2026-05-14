import os
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

warnings.filterwarnings("ignore")

DATASET_PATH = os.path.join("..", "Data", "Healthcare.csv")
GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}

HIGH_RISK = {
    "Heart Disease", "Stroke", "Diabetes", "Hypertension",
    "Chronic Kidney Disease", "Liver Disease", "Epilepsy",
    "Parkinson's", "Tuberculosis", "Pneumonia", "COVID-19",
}
MEDIUM_RISK = {
    "Asthma", "Thyroid Disorder", "Anemia", "Depression", "Obesity",
    "Arthritis", "Dementia", "IBS", "Ulcer", "Bronchitis",
    "Sinusitis", "Gastritis", "Anxiety", "Food Poisoning", "Influenza",
}

RISK_COLORS = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#27ae60"}


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()
    df.drop(columns=["Patient_ID"], inplace=True, errors="ignore")
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Gender"].fillna("Other", inplace=True)
    df["Symptoms"].fillna("", inplace=True)
    df.dropna(subset=["Disease"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Gender_enc"] = df["Gender"].map(GENDER_MAP).fillna(2).astype(int)
    df["Symptoms_list"] = df["Symptoms"].apply(
        lambda s: [sym.strip().lower() for sym in str(s).split(",") if sym.strip()]
    )

    mlb = MultiLabelBinarizer()
    sym_matrix = mlb.fit_transform(df["Symptoms_list"])
    sym_df = pd.DataFrame(sym_matrix, columns=mlb.classes_)

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df["Disease"])

    base = df[["Age", "Gender_enc"]].reset_index(drop=True)
    X = pd.concat([base, sym_df], axis=1).values.astype(float)

    print(f"[INFO] Features : {X.shape[1]}  (2 base + {len(mlb.classes_)} symptoms)")
    print(f"[INFO] Classes  : {len(label_enc.classes_)}")
    return X, y, mlb, label_enc, sorted(mlb.classes_.tolist())


def train_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    nb = GaussianNB()
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)

    ensemble = VotingClassifier(
        estimators=[
            ("Random Forest", rf),
            ("Logistic Regression", lr),
            ("Naive Bayes", nb),
            ("SVM", svm),
        ],
        voting="soft",
        n_jobs=-1,
    )

    print("[INFO] Training individual models + ensemble (this may take ~30 s) ...")
    ensemble.fit(X_train, y_train)
    print("[INFO] Training complete.")

    fitted_rf = ensemble.estimators_[0]
    fitted_lr = ensemble.estimators_[1]
    fitted_nb = ensemble.estimators_[2]
    fitted_svm = ensemble.estimators_[3]

    return ensemble, fitted_rf, fitted_lr, fitted_nb, fitted_svm


def evaluate_models(ensemble, rf, lr, nb, svm, X_test, y_test):
    print(f"\n{'='*54}")
    print("  Model Accuracy Comparison")
    print(f"{'='*54}")

    results = [
        ("Random Forest", rf),
        ("Logistic Regression", lr),
        ("Naive Bayes", nb),
        ("SVM", svm),
        ("Ensemble (Voting)", ensemble),
    ]

    for name, model in results:
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        marker = "  ◄ ENSEMBLE" if "Ensemble" in name else ""
        print(f"  {name:<26} Accuracy: {acc:6.2f}%{marker}")

    cm = confusion_matrix(y_test, ensemble.predict(X_test))
    print(f"\n  Ensemble Confusion Matrix ({cm.shape[0]}×{cm.shape[1]}):")
    print(cm)
    print("=" * 54)


def predict_disease(ensemble, mlb: MultiLabelBinarizer, label_enc: LabelEncoder,
                    age: float, gender: str, symptoms_str: str):
    gender_val = GENDER_MAP.get(gender, 2)
    sym_list = [s.strip().lower() for s in symptoms_str.split(",") if s.strip()]
    sym_vec = mlb.transform([sym_list])

    base = np.array([[float(age), gender_val]])
    X = np.hstack([base, sym_vec]).astype(float)

    pred_enc = ensemble.predict(X)[0]
    disease = label_enc.inverse_transform([pred_enc])[0]

    proba = ensemble.predict_proba(X)[0]
    confidence = round(float(proba.max()) * 100, 1)

    if disease in HIGH_RISK:
        risk = "High"
    elif disease in MEDIUM_RISK:
        risk = "Medium"
    else:
        risk = "Low"

    return disease, confidence, risk


class App:
    def __init__(self, root: tk.Tk, ensemble, mlb, label_enc, symptom_list):
        self.root = root
        self.ensemble = ensemble
        self.mlb = mlb
        self.label_enc = label_enc
        self.symptom_list = symptom_list

        self.root.title("Disease Prediction System")
        self.root.geometry("560x680")
        self.root.resizable(False, False)
        self.root.configure(bg="#f5f5f5")

        self._build_ui()

    def _build_ui(self):
        tk.Label(
            self.root, text="Disease Prediction System",
            font=("Helvetica", 15, "bold"),
            bg="#2c3e50", fg="white", pady=10
        ).pack(fill="x")

        canvas = tk.Canvas(self.root, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.main_frame = tk.Frame(canvas, bg="#f5f5f5", padx=24, pady=16)
        canvas_window = canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(e):
            canvas.itemconfig(canvas_window, width=e.width)

        self.main_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        form = self.main_frame

        self._lbl(form, "Age:", 0)
        self.age_var = tk.StringVar()
        tk.Entry(form, textvariable=self.age_var, width=34).grid(row=0, column=1, pady=6, sticky="w")

        self._lbl(form, "Gender:", 1)
        self.gender_var = tk.StringVar(value="Male")
        ttk.Combobox(
            form, textvariable=self.gender_var,
            values=["Male", "Female", "Other"],
            state="readonly", width=32
        ).grid(row=1, column=1, pady=6, sticky="w")

        tk.Label(
            form, text="Symptoms (select all that apply):",
            font=("Helvetica", 10, "bold"), bg="#f5f5f5"
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 4))

        sym_frame = tk.Frame(form, bg="#f5f5f5")
        sym_frame.grid(row=3, column=0, columnspan=2, sticky="w")

        self.sym_vars = {}
        cols = 3
        for idx, sym in enumerate(self.symptom_list):
            var = tk.IntVar()
            self.sym_vars[sym] = var
            tk.Checkbutton(
                sym_frame, text=sym.title(), variable=var,
                bg="#f5f5f5", anchor="w", width=20
            ).grid(row=idx // cols, column=idx % cols, sticky="w")

        tk.Label(
            form, text="Or type symptoms (comma-separated):",
            font=("Helvetica", 9), fg="#555", bg="#f5f5f5"
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 2))

        self.sym_text_var = tk.StringVar()
        tk.Entry(form, textvariable=self.sym_text_var, width=46).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))

        tk.Label(
            form, text="e.g.  fever, cough, headache, fatigue",
            font=("Helvetica", 8), fg="#999", bg="#f5f5f5"
        ).grid(row=6, column=0, columnspan=2, sticky="w")

        tk.Button(
            form, text="  Predict Disease  ",
            command=self._on_predict,
            bg="#2980b9", fg="white",
            font=("Helvetica", 11, "bold"),
            padx=10, pady=6, relief="flat", cursor="hand2"
        ).grid(row=7, column=0, columnspan=2, pady=16)

        res = tk.Frame(form, bg="#ecf0f1", padx=14, pady=10, relief="groove", bd=1)
        res.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        tk.Label(res, text="Prediction Result",
                 font=("Helvetica", 10, "bold"), bg="#ecf0f1").pack(anchor="w")

        self.disease_lbl = tk.Label(
            res, text="—",
            font=("Helvetica", 13, "bold"),
            bg="#ecf0f1", fg="#2c3e50"
        )
        self.disease_lbl.pack(anchor="w", pady=(4, 0))

        self.conf_lbl = tk.Label(res, text="", font=("Helvetica", 10), bg="#ecf0f1", fg="#555")
        self.conf_lbl.pack(anchor="w")

        self.risk_lbl = tk.Label(res, text="", font=("Helvetica", 11, "bold"), bg="#ecf0f1")
        self.risk_lbl.pack(anchor="w")

    def _lbl(self, parent, text, row):
        tk.Label(
            parent, text=text,
            font=("Helvetica", 10), bg="#f5f5f5", anchor="e"
        ).grid(row=row, column=0, sticky="e", padx=(0, 10), pady=6)

    def _on_predict(self):
        age_str = self.age_var.get().strip()

        if not age_str:
            messagebox.showwarning("Missing Input", "Please enter Age.")
            return
        try:
            age = float(age_str)
            if not (0 < age < 121):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Age must be a number between 1 and 120.")
            return

        checked = [s for s, v in self.sym_vars.items() if v.get() == 1]
        typed = [s.strip().lower() for s in self.sym_text_var.get().split(",") if s.strip()]
        all_syms = list(dict.fromkeys(checked + typed))

        if not all_syms:
            messagebox.showwarning("Missing Input", "Please select or type at least one symptom.")
            return

        symptoms_str = ", ".join(all_syms)

        try:
            disease, confidence, risk = predict_disease(
                self.ensemble, self.mlb, self.label_enc,
                age, self.gender_var.get(), symptoms_str
            )
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))
            return

        self.disease_lbl.config(text=f"Predicted Disease:  {disease}")
        self.conf_lbl.config(text=f"Confidence Score:  {confidence}%")
        self.risk_lbl.config(text=f"Risk Level:  {risk}", fg=RISK_COLORS.get(risk, "#2c3e50"))

        if risk == "High":
            messagebox.showwarning(
                "⚠  High Risk Alert",
                f"'{disease}' is classified as HIGH RISK.\n"
                "Please consult a medical professional immediately."
            )


def main():
    print("\n" + "=" * 54)
    print("  Disease Prediction System — Starting Up")
    print("=" * 54)

    df = load_data(DATASET_PATH)
    X, y, mlb, label_enc, symptom_list = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}  |  Test: {len(X_test)}")

    ensemble, rf, lr, nb, svm = train_model(X_train, y_train)
    evaluate_models(ensemble, rf, lr, nb, svm, X_test, y_test)

    print("\n[INFO] Launching GUI ...")
    root = tk.Tk()
    App(root, ensemble, mlb, label_enc, symptom_list)
    root.mainloop()


if __name__ == "__main__":
    main()
