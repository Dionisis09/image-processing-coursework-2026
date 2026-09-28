import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORT_PATH = ROOT_DIR / "Texniki_Anafora_Ikona_2025_2026_v4.docx"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def writable_report_path(base_path: Path) -> Path:
    """Return a writable docx path, falling back to versioned names if needed."""
    if not base_path.exists():
        return base_path

    try:
        with open(base_path, "ab"):
            return base_path
    except PermissionError:
        for idx in range(2, 20):
            candidate = base_path.with_stem(f"{base_path.stem}_v{idx}")
            if not candidate.exists():
                return candidate
            try:
                with open(candidate, "ab"):
                    return candidate
            except PermissionError:
                continue
        raise PermissionError(f"No writable output path available near {base_path}")


def add_picture_if_exists(document: Document, path: Path, width: float = 6.3) -> None:
    if path.exists():
        document.add_picture(str(path), width=Inches(width))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_result_block(
    document: Document,
    title: str,
    path: Path,
    command: str,
    display_note: str,
    caption: str,
    width: float = 6.3,
) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(title + "\n")
    run.bold = True
    paragraph.add_run(f"Εντολή εκτέλεσης: {command}\n")
    paragraph.add_run(f"Αρχείο αποτελέσματος: {rel_path(path)}\n")
    paragraph.add_run(f"Εμφάνιση στην εξέταση: {display_note}")
    add_picture_if_exists(document, path, width=width)
    add_caption(document, caption)


def add_caption(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.runs[0].italic = True


def add_metric_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers):
        cell.text = header

    for row_values in rows:
        row = table.add_row().cells
        for cell, value in zip(row, row_values):
            cell.text = value


def exercise_1_section(document: Document) -> None:
    cmd = "python askisi1_erotima1.py"
    document.add_heading("1. Φιλτράρισμα στο Πεδίο Συχνοτήτων", level=1)
    document.add_paragraph(
        "Για την εικόνα moon.jpg υλοποιήθηκε γραμμικός μετασχηματισμός δυναμικής περιοχής, "
        "μεταφορά του σημείου (0,0) στο κέντρο με την ιδιότητα του DFT, 2D-DFT με μέθοδο "
        "γραμμών-στηλών, απεικόνιση του φάσματος και εφαρμογή ιδανικού lowpass φίλτρου."
    )
    document.add_paragraph(
        "Εξετάστηκαν ακτίνες αποκοπής 15, 25, 35, 45 και 60 pixels. Επιλέχθηκε η τιμή 35 "
        "ως ικανοποιητικός συμβιβασμός μεταξύ εξομάλυνσης και διατήρησης δομής."
    )
    add_result_block(
        document,
        "Αποτέλεσμα 1: Προεπεξεργασία της εικόνας",
        OUTPUTS_DIR / "askisi1_erotima1" / "preprocessing_summary.png",
        cmd,
        "Ανοίγουμε το PNG από τον φάκελο outputs ή το δείχνουμε με matplotlib ως συνοπτική σύνθετη εικόνα.",
        "Σχήμα 1. Προεπεξεργασία της εικόνας moon.jpg.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 2: Μέτρο του 2D-DFT σε linear και log scale",
        OUTPUTS_DIR / "askisi1_erotima1" / "spectrum_summary.png",
        cmd,
        "Προβάλλουμε την αποθηκευμένη σύνθετη εικόνα του φάσματος για να δείξουμε και τις δύο κλίμακες.",
        "Σχήμα 2. Μέτρο του 2D-DFT σε linear και log scale.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 3: Lowpass filtering και αποκατάσταση",
        OUTPUTS_DIR / "askisi1_erotima1" / "filtering_summary.png",
        cmd,
        "Εμφανίζουμε τη σύνθετη εικόνα με τη μάσκα, το φιλτραρισμένο φάσμα και την τελική εικόνα.",
        "Σχήμα 3. Ιδανικό lowpass φίλτρο, φιλτραρισμένο φάσμα και τελική εικόνα.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 4: Επιλογή ακτίνας αποκοπής",
        OUTPUTS_DIR / "askisi1_erotima1" / "cutoff_mse_curve.png",
        cmd,
        "Προβάλλουμε την καμπύλη MSE ως προς την ακτίνα αποκοπής για να αιτιολογήσουμε την επιλογή D0=35.",
        "Σχήμα 4. MSE ως προς την ακτίνα αποκοπής.",
    )
    document.add_paragraph(
        "Σχολιασμός: Η γραμμική απεικόνιση του φάσματος αναδεικνύει κυρίως την πολύ ισχυρή "
        "συνιστώσα χαμηλών συχνοτήτων, ενώ η λογαριθμική απεικόνιση επιτρέπει την παρατήρηση "
        "και ασθενέστερων συνιστωσών. Το lowpass φίλτρο μειώνει τις υψηλές συχνότητες, άρα "
        "η τελική εικόνα γίνεται ομαλότερη αλλά εμφανίζει απώλεια οξύτητας στις λεπτομέρειες."
    )


def exercise_2_section(document: Document) -> None:
    summary_text = (OUTPUTS_DIR / "askisi2_histogram_equalization" / "summary.txt").read_text(encoding="utf-8")
    cmd = "python askisi2_histogram_equalization.py"
    document.add_heading("2. Βελτίωση Εικόνας - Εξίσωση Ιστογράμματος", level=1)
    document.add_paragraph(
        "Εφαρμόστηκε ολική εξίσωση ιστογράμματος και τοπική προσαρμοστική εξίσωση "
        "ιστογράμματος (CLAHE) στις τρεις εικόνες dark_road. Εξετάστηκαν παράθυρα "
        "32x32, 64x64 και 128x128, με τελική επιλογή 64x64."
    )
    document.add_paragraph(
        "Συνοπτικά αποτελέσματα: επιλέχθηκε παράθυρο 64x64 για την προσαρμοστική εξίσωση, "
        "καθώς προσφέρει καλύτερη ισορροπία ανάμεσα στην ενίσχυση της αντίθεσης και στον "
        "περιορισμό της ενίσχυσης θορύβου. Οι τρεις εικόνες παρουσιάζουν σαφή αύξηση της "
        "διασποράς των γκρι τιμών μετά την ολική εξίσωση, ενώ η τοπική εξίσωση διατηρεί πιο "
        "ελεγχόμενη συμπεριφορά."
    )
    for image_name in ("dark_road_1", "dark_road_2", "dark_road_3"):
        add_result_block(
            document,
            f"Αποτέλεσμα: Ιστόγραμμα, ολική και τοπική εξίσωση ιστογράμματος για {image_name}",
            OUTPUTS_DIR / "askisi2_histogram_equalization" / f"{image_name}_histograms.png",
            cmd,
            "Ανοίγουμε τη σύνθετη εικόνα PNG που δείχνει μαζί τις εικόνες και τα ιστογράμματα.",
            f"Σχήμα. Ιστόγραμμα και αποτελέσματα για {image_name}.",
        )
        add_result_block(
            document,
            f"Αποτέλεσμα: Σύγκριση παραθύρων CLAHE για {image_name}",
            OUTPUTS_DIR / "askisi2_histogram_equalization" / f"{image_name}_adaptive_candidates.png",
            cmd,
            "Προβάλλουμε τη σύνθετη εικόνα με παράθυρα 32x32, 64x64 και 128x128 για να αιτιολογηθεί το 64x64.",
            f"Σχήμα. Σύγκριση υποψήφιων παραθύρων CLAHE για {image_name}.",
        )
    document.add_paragraph(
        "Σχολιασμός: Η ολική εξίσωση ιστογράμματος αυξάνει έντονα την αντίθεση αλλά συχνά "
        "παράγει υπερενίσχυση σε πολύ σκοτεινές περιοχές. Η τοπική εξίσωση ιστογράμματος "
        "(CLAHE) βελτιώνει πιο ελεγχόμενα την ορατότητα των λεπτομερειών, όμως πολύ μικρό "
        "παράθυρο τείνει να ενισχύει και θόρυβο. Για αυτό επιλέχθηκε παράθυρο 64x64."
    )


def exercise_3_section(document: Document) -> None:
    summary = load_json(OUTPUTS_DIR / "askisi3_denoising" / "summary.json")
    cmd = "python askisi3_denoising.py"
    document.add_heading("3. Αποθορυβοποίηση Εικόνας", level=1)
    document.add_paragraph(
        "Στην εικόνα tiger.mat προστέθηκε Gaussian θόρυβος με τελικό SNR 15 dB, "
        "κρουστικός θόρυβος 20%, καθώς και συνδυασμός των δύο. Χρησιμοποιήθηκαν "
        "moving average και median filters με kernels 3x3 και 5x5."
    )
    add_metric_table(
        document,
        ["Περίπτωση", "Μέθοδος", "MSE", "PSNR (dB)"],
        [
            ["Gaussian", "Noisy", f"{summary['gaussian_case']['noisy_metrics']['mse']:.6f}", f"{summary['gaussian_case']['noisy_metrics']['psnr_db']:.3f}"],
            ["Gaussian", "Median 3x3", f"{summary['gaussian_case']['filters']['median_3']['mse']:.6f}", f"{summary['gaussian_case']['filters']['median_3']['psnr_db']:.3f}"],
            ["Impulse", "Noisy", f"{summary['impulse_case']['noisy_metrics']['mse']:.6f}", f"{summary['impulse_case']['noisy_metrics']['psnr_db']:.3f}"],
            ["Impulse", "Median 3x3", f"{summary['impulse_case']['filters']['median_3']['mse']:.6f}", f"{summary['impulse_case']['filters']['median_3']['psnr_db']:.3f}"],
            ["Mixed", "Mean -> Median", f"{summary['combined_case']['mean_then_median']['mse']:.6f}", f"{summary['combined_case']['mean_then_median']['psnr_db']:.3f}"],
            ["Mixed", "Median -> Mean", f"{summary['combined_case']['median_then_mean']['mse']:.6f}", f"{summary['combined_case']['median_then_mean']['psnr_db']:.3f}"],
        ],
    )
    document.add_paragraph(
        f"Το πραγματικό SNR της Gaussian υποβάθμισης ρυθμίστηκε ακριβώς σε {summary['gaussian_actual_snr_db']:.6f} dB. "
        f"Για τον μεικτό θόρυβο προτιμήθηκε η σειρά {summary['combined_case']['recommended_order']}."
    )
    add_result_block(
        document,
        "Αποτέλεσμα 1: Σύγκριση φίλτρων για Gaussian θόρυβο",
        OUTPUTS_DIR / "askisi3_denoising" / "gaussian_kernel_comparison.png",
        cmd,
        "Προβάλλουμε τη σύνθετη εικόνα με τη θορυβώδη είσοδο και τα αποτελέσματα mean/median για 3x3 και 5x5.",
        "Σχήμα 1. Σύγκριση φίλτρων για Gaussian θόρυβο.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 2: Σύγκριση φίλτρων για κρουστικό θόρυβο",
        OUTPUTS_DIR / "askisi3_denoising" / "impulse_kernel_comparison.png",
        cmd,
        "Ανοίγουμε την αντίστοιχη σύνθετη εικόνα PNG για να φανεί η υπεροχή του median filter.",
        "Σχήμα 2. Σύγκριση φίλτρων για κρουστικό θόρυβο.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 3: Σειρά φιλτραρίσματος για μεικτό θόρυβο",
        OUTPUTS_DIR / "askisi3_denoising" / "combined_median_then_mean.png",
        cmd,
        "Προβάλλουμε την καλύτερη σειρά median -> mean και σχολιάζουμε το τελικό αποτέλεσμα.",
        "Σχήμα 3. Βέλτιστη σειρά φιλτραρίσματος για μεικτό θόρυβο.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 4: Εναλλακτική σειρά mean -> median",
        OUTPUTS_DIR / "askisi3_denoising" / "combined_mean_then_median.png",
        cmd,
        "Το εμφανίζουμε για ποιοτική σύγκριση με τη βέλτιστη σειρά.",
        "Σχήμα 4. Εναλλακτική σειρά φιλτραρίσματος για μεικτό θόρυβο.",
    )
    document.add_paragraph(
        "Σχολιασμός: Για Gaussian θόρυβο, το moving average και το median με μικρό kernel "
        "δίνουν παρόμοια συμπεριφορά, με ελαφρά υπεροχή του median 3x3 στο συγκεκριμένο παράδειγμα. "
        "Για impulse noise, το median filter είναι σαφώς καταλληλότερο, επειδή απορρίπτει "
        "τις ακραίες τιμές χωρίς να θολώνει τόσο πολύ τις ακμές. Στον μεικτό θόρυβο, η "
        "σειρά median -> mean είναι πιο αποτελεσματική, καθώς πρώτα απομακρύνει τα impulses "
        "και έπειτα εξομαλύνει τον Gaussian θόρυβο."
    )


def exercise_4_section(document: Document) -> None:
    summary = load_json(OUTPUTS_DIR / "askisi4_restoration" / "summary.json")
    cmd = "python askisi4_restoration.py"
    document.add_heading("4. Αποκατάσταση Εικόνας - Wiener και Inverse Filtering", level=1)
    document.add_paragraph(
        "Στο Μέρος Α προστέθηκε Gaussian θόρυβος με τελικό SNR 10 dB και εφαρμόστηκε "
        "Wiener filtering στο πεδίο συχνοτήτων για δύο περιπτώσεις: γνωστή ισχύς θορύβου "
        "και εκτιμώμενη ισχύς θορύβου."
    )
    add_metric_table(
        document,
        ["Περίπτωση", "MSE", "PSNR (dB)"],
        [
            ["Noisy", f"{summary['part_a_metrics']['noisy']['mse']:.3f}", f"{summary['part_a_metrics']['noisy']['psnr_db']:.3f}"],
            ["Wiener - Known Noise", f"{summary['part_a_metrics']['known_noise_wiener']['mse']:.3f}", f"{summary['part_a_metrics']['known_noise_wiener']['psnr_db']:.3f}"],
            ["Wiener - Unknown Noise", f"{summary['part_a_metrics']['unknown_noise_wiener']['mse']:.3f}", f"{summary['part_a_metrics']['unknown_noise_wiener']['psnr_db']:.3f}"],
        ],
    )
    document.add_paragraph(
        f"Το πραγματικό SNR της υποβαθμισμένης εικόνας είναι {summary['actual_snr_db']:.6f} dB. "
        f"Η εκτιμώμενη ισχύς θορύβου στην άγνωστη περίπτωση ήταν {summary['estimated_noise_power_unknown_case']:.3f}."
    )
    add_result_block(
        document,
        "Αποτέλεσμα 1: Wiener filtering",
        OUTPUTS_DIR / "askisi4_restoration" / "wiener_summary.png",
        cmd,
        "Εμφανίζουμε τη σύνθετη εικόνα με original, noisy, Wiener με γνωστή ισχύ θορύβου και Wiener με εκτιμώμενη ισχύ θορύβου.",
        "Σχήμα 1. Αποτελέσματα Wiener filtering.",
    )
    document.add_paragraph(
        "Στο Μέρος Β χρησιμοποιήθηκε το psf.p μέσω MATLAB για εξαγωγή της κρουστικής απόκρισης "
        "και της θολωμένης εικόνας. Στη συνέχεια εφαρμόστηκε inverse filtering με threshold."
    )
    add_result_block(
        document,
        "Αποτέλεσμα 2: Θολωμένη εικόνα από το psf.p",
        OUTPUTS_DIR / "askisi4_restoration" / "blurred_from_psf.png",
        cmd,
        "Προβάλλουμε την εικόνα που παράχθηκε από το MATLAB export του psf.p.",
        "Σχήμα 2. Θολωμένη εικόνα μετά την εφαρμογή του psf.p.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 3: Εκτιμημένη κρουστική απόκριση",
        OUTPUTS_DIR / "askisi4_restoration" / "estimated_impulse_response.png",
        cmd,
        "Ανοίγουμε το PNG της impulse response που εξήχθη από το MATLAB helper script.",
        "Σχήμα 3. Εκτιμημένη κρουστική απόκριση του αγνώστου συστήματος.",
        width=3.0,
    )
    add_result_block(
        document,
        "Αποτέλεσμα 4: Log απόκριση συχνότητας",
        OUTPUTS_DIR / "askisi4_restoration" / "frequency_response_log.png",
        cmd,
        "Προβάλλουμε τη log απεικόνιση της απόκρισης συχνότητας του εκτιμημένου συστήματος.",
        "Σχήμα 4. Log απεικόνιση της απόκρισης συχνότητας.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 5: Καμπύλη MSE ως προς το threshold",
        OUTPUTS_DIR / "askisi4_restoration" / "inverse_filter_threshold_curve.png",
        cmd,
        "Εμφανίζουμε την καμπύλη threshold sweep για να αιτιολογήσουμε την επιλογή βέλτιστου threshold.",
        "Σχήμα 5. MSE ως προς το threshold του inverse filter.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 6: Inverse filtering χωρίς threshold",
        OUTPUTS_DIR / "askisi4_restoration" / "inverse_filter_no_threshold.png",
        cmd,
        "Προβάλλουμε το no-threshold αποτέλεσμα για άμεση ποιοτική σύγκριση.",
        "Σχήμα 6. Αποτέλεσμα inverse filtering χωρίς threshold.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 7: Βέλτιστο inverse filtering",
        OUTPUTS_DIR / "askisi4_restoration" / "inverse_filter_best_restored.png",
        cmd,
        "Ανοίγουμε το καλύτερο restored αποτέλεσμα που προέκυψε από το sweep των thresholds.",
        "Σχήμα 7. Καλύτερο αποτέλεσμα inverse filtering.",
    )
    document.add_paragraph(
        f"Βέλτιστο threshold: {summary['part_b_inverse_filter']['best_threshold']:.6f}. "
        f"Best MSE: {summary['part_b_inverse_filter']['best_mse']:.6f}. "
        f"No-threshold MSE: {summary['part_b_inverse_filter']['no_threshold_mse']:.6f}."
    )
    document.add_paragraph(
        "Χωρίς threshold, το inverse filter χρησιμοποιεί όλες τις συχνοτικές συνιστώσες. "
        "Στο συγκεκριμένο σύστημα οι μικρότερες τιμές της απόκρισης συχνότητας δεν είναι "
        "εξαιρετικά κοντά στο μηδέν, άρα η αστάθεια είναι περιορισμένη, όμως η χρήση "
        "threshold εξακολουθεί να βελτιώνει ελαφρά το τελικό σφάλμα."
    )
    document.add_paragraph(
        "Σχολιασμός: Το Wiener filtering μειώνει σημαντικά τον προστιθέμενο θόρυβο, με την "
        "εκτιμώμενη ισχύ θορύβου να αποδίδει ελαφρώς καλύτερα στο συγκεκριμένο παράδειγμα. "
        "Στο inverse filtering, η χρήση threshold περιορίζει την ενίσχυση συνιστωσών όπου "
        "η απόκριση του συστήματος είναι ασθενής. Η βέλτιστη τιμή threshold δίνει μικρότερο "
        "MSE από την περίπτωση χωρίς threshold, ακόμη κι αν εδώ η διαφορά είναι σχετικά μικρή."
    )


def exercise_5_section(document: Document) -> None:
    summary = load_json(OUTPUTS_DIR / "askisi5_dct_compression" / "summary.json")
    cmd = "python askisi5_dct_compression.py"
    document.add_heading("5. Συμπίεση Εικόνας με χρήση 2D-DCT", level=1)
    document.add_paragraph(
        "Η εικόνα board.png τεμαχίστηκε σε blocks 32x32. Για κάθε block εφαρμόστηκε 2D-DCT "
        "και δοκιμάστηκαν δύο μέθοδοι διατήρησης συντελεστών: zonal coding και threshold coding."
    )
    add_metric_table(
        document,
        ["r (%)", "Zonal MSE", "Threshold MSE"],
        [
            [str(r), f"{summary['metrics']['zonal'][str(r)]['mse']:.4f}", f"{summary['metrics']['threshold'][str(r)]['mse']:.4f}"]
            for r in summary["r_values"]
        ],
    )
    add_result_block(
        document,
        "Αποτέλεσμα 1: Ενδεικτικές ανακατασκευές με zonal coding",
        OUTPUTS_DIR / "askisi5_dct_compression" / "zonal_showcase.png",
        cmd,
        "Προβάλλουμε τη σύνθετη εικόνα με r=10%, 25%, 50% για zonal coding.",
        "Σχήμα 1. Ενδεικτικές ανακατασκευές με zonal coding.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 2: Ενδεικτικές ανακατασκευές με threshold coding",
        OUTPUTS_DIR / "askisi5_dct_compression" / "threshold_showcase.png",
        cmd,
        "Προβάλλουμε την αντίστοιχη σύνθετη εικόνα για threshold coding.",
        "Σχήμα 2. Ενδεικτικές ανακατασκευές με threshold coding.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 3: Καμπύλες MSE ως προς r",
        OUTPUTS_DIR / "askisi5_dct_compression" / "mse_curve.png",
        cmd,
        "Ανοίγουμε το διάγραμμα MSE για σύγκριση των δύο μεθόδων συμπίεσης.",
        "Σχήμα 3. Καμπύλες MSE ως προς το ποσοστό διατήρησης πληροφορίας.",
    )
    document.add_paragraph(
        "Σχολιασμός: Καθώς αυξάνεται το ποσοστό διατήρησης πληροφορίας r, το MSE μειώνεται "
        "μονοτονικά και για τις δύο μεθόδους. Η threshold coding υπερέχει ποσοτικά σε όλες "
        "τις τιμές του r, επειδή κρατά τους ενεργειακά σημαντικότερους συντελεστές κάθε block, "
        "ενώ η zonal coding επιβάλλει σταθερό γεωμετρικό πρότυπο επιλογής."
    )


def exercise_6_section(document: Document) -> None:
    summary = load_json(OUTPUTS_DIR / "askisi6_edges" / "summary.json")
    cmd = "python askisi6_edges.py"
    document.add_heading("6. Ανίχνευση Ακμών", level=1)
    document.add_paragraph(
        "Χρησιμοποιήθηκαν οι Sobel μάσκες για τον υπολογισμό των χωρικών παραγώγων, "
        "υπολογίστηκε το gradient magnitude και εφαρμόστηκε κατωφλίωση με Otsu threshold."
    )
    document.add_paragraph(
        f"Το επιλεγμένο threshold ήταν T = {summary['threshold_value']:.1f}. "
        f"Ανιχνεύθηκαν {summary['detected_line_segments']} line segments με probabilistic Hough transform."
    )
    add_result_block(
        document,
        "Αποτέλεσμα 1: Sobel αποκρίσεις",
        OUTPUTS_DIR / "askisi6_edges" / "sobel_summary.png",
        cmd,
        "Προβάλλουμε τη σύνθετη εικόνα με την αρχική εικόνα και τις Sobel αποκρίσεις σε x και y.",
        "Σχήμα 1. Αποτελέσματα Sobel σε x και y διεύθυνση.",
    )
    add_result_block(
        document,
        "Αποτέλεσμα 2: Gradient, thresholding και γραμμές",
        OUTPUTS_DIR / "askisi6_edges" / "edges_and_lines_summary.png",
        cmd,
        "Εμφανίζουμε την τελική σύνθετη εικόνα με magnitude, edge map και επικάλυψη ανιχνευμένων γραμμών.",
        "Σχήμα 2. Gradient magnitude, edge map και overlay γραμμών.",
    )
    document.add_paragraph(
        "Σχολιασμός: Οι Sobel μάσκες αναδεικνύουν τις απότομες μεταβολές φωτεινότητας, ενώ "
        "το gradient magnitude δίνει ενιαία μέτρηση ισχύος ακμής. Η κατωφλίωση με Otsu "
        "παρήγαγε ένα λειτουργικό edge map και ο probabilistic Hough transform επέτρεψε την "
        "επισήμανση των κυρίαρχων ευθύγραμμων τμημάτων της σκηνής."
    )


def appendix_section(document: Document) -> None:
    document.add_heading("Παράρτημα - Περιβάλλον Υλοποίησης", level=1)
    document.add_paragraph(
        "Η υλοποίηση πραγματοποιήθηκε σε Python 3.13. Χρησιμοποιήθηκαν κυρίως οι βιβλιοθήκες "
        "NumPy, SciPy, scikit-image, Pillow, Matplotlib και python-docx."
    )
    document.add_paragraph(
        "Για την Άσκηση 4 και ειδικά για το αρχείο psf.p χρησιμοποιήθηκε MATLAB R2026a, "
        "καθώς το συγκεκριμένο αρχείο είναι MATLAB p-code και δεν εκτελείται απευθείας από Python."
    )
    document.add_paragraph(
        "Η εκτέλεση όλων των scripts μπορεί να γίνει συνολικά με την εντολή: "
        "python run_all_assignments.py"
    )


def main() -> None:
    document = Document()
    normal_style = document.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Τεχνική Αναφορά\nΨηφιακή Επεξεργασία και Ανάλυση Εικόνας - Μέρος Α")
    run.bold = True
    run.font.size = Pt(16)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Ακαδημαϊκό Έτος 2025-2026\nΥλοποίηση σε Python").italic = True

    cover = document.add_paragraph()
    cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cover.add_run(
        "\nΟνοματεπώνυμο: \n"
        "Αριθμός Μητρώου: \n"
        "Μάθημα: Ψηφιακή Επεξεργασία και Ανάλυση Εικόνας\n"
        "Τμήμα: Μηχανικών Η/Υ και Πληροφορικής\n"
        "Ημερομηνία Παράδοσης: "
    )

    document.add_paragraph(
        "Η παρούσα αναφορά συνοδεύει την υλοποίηση των ζητούμενων ασκήσεων του PDF και "
        "περιλαμβάνει περιγραφή της μεθοδολογίας, ενδεικτικά αποτελέσματα και βασικά "
        "ποσοτικά metrics (MSE, PSNR όπου απαιτούνται)."
    )
    document.add_paragraph(
        "Σημείωση παρουσίασης: πριν από κάθε σχήμα αναγράφεται η εντολή εκτέλεσης, το αρχείο "
        "αποτελέσματος και ο προτεινόμενος τρόπος προβολής, ώστε να μπορεί να χρησιμοποιηθεί "
        "άμεσα και κατά την προφορική/εργαστηριακή εξέταση."
    )

    exercise_1_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    exercise_2_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    exercise_3_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    exercise_4_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    exercise_5_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    exercise_6_section(document)
    document.add_section(WD_SECTION_START.NEW_PAGE)
    appendix_section(document)

    output_path = writable_report_path(REPORT_PATH)
    document.save(output_path)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
