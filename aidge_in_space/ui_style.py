"""! @brief Apparence de l'interface : thème Gradio, feuille de style et blocs de présentation communs aux onglets."""
from contextlib import contextmanager

import gradio as gr

## Polices du système : lisibles, et aucune dépendance réseau sur une machine hors ligne.
FONT = ("Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif")
## Police à chasse fixe du journal et du code.
FONT_MONO = ("Cascadia Mono", "Consolas", "ui-monospace", "Menlo", "monospace")

## Thème passé à launch() : libellés sobres à la place des étiquettes colorées de Soft.
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", font=FONT, font_mono=FONT_MONO).set(
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_title_text_color="*body_text_color",
    block_title_text_color_dark="*body_text_color",
    block_title_text_weight="600",
    block_title_padding="0",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="*body_text_color_subdued",
    block_label_text_color_dark="*body_text_color_subdued",
    block_label_text_weight="600",
    input_background_fill="*neutral_50",
    input_border_color="*neutral_200",
    input_border_width="1px",
    input_shadow="none",
    button_primary_text_color="white",
)

## Feuille de style passée à launch(). Les couleurs viennent des variables du thème pour suivre le mode sombre.
CSS = """
.block.hero-block, .hero-block .html-container { padding: 0 !important; margin: 0 !important; }
.hero {
    padding: 18px 24px; border-radius: 14px; color: #e8ecf8;
    background: radial-gradient(circle at 92% 10%, #3b4a8f 0%, transparent 42%), linear-gradient(120deg, #111a3a, #1d2756);
}
.hero h1 { margin: 0 0 4px; font-size: 1.5rem; font-weight: 700; line-height: 1.25; color: #fff; }
.hero .eyebrow { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em; color: #aab6ec; }
.hero p { margin: 0; color: #cdd5f3; }
.hero ol { display: flex; flex-wrap: wrap; gap: 4px 22px; margin: 10px 0 0; padding: 0; list-style: none; font-size: 0.875rem; }
.hero li { margin: 0; color: #cdd5f3; }
.hero li b { color: #fff; font-weight: 600; }

.block.tab-intro {
    padding: 10px 16px !important; border-left: 3px solid var(--color-accent); border-radius: 0 8px 8px 0;
    background: var(--background-fill-secondary);
}
.prose.tab-intro p { margin: 0 0 2px !important; }
.prose.tab-intro p:last-child { margin-bottom: 0 !important; }

.card {
    padding: 16px !important; gap: 14px !important; border-radius: 12px;
    border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
}
.card > .block:not(.gr-accordion), .card .form, .card .form > .block {
    padding: 0 !important; border: none !important; box-shadow: none !important; background: transparent !important;
}
.card .form { gap: 14px !important; }

.block.gr-accordion { border: 1px solid var(--border-color-primary) !important; border-radius: 10px; box-shadow: none; }

.prose.card-title h3 { margin: 0 !important; font-size: 1rem; font-weight: 700; }
.prose.card-title p { margin: 2px 0 0 !important; font-size: 0.875rem; color: var(--body-text-color-subdued); }

.block.hint { padding: 0 !important; }
.prose.hint p, .prose.hint li { margin: 0 !important; font-size: 0.875rem; color: var(--body-text-color-subdued); }

.block.status-banner {
    padding: 10px 16px !important; border-radius: 8px; border: 1px solid var(--border-color-primary);
    background: var(--block-background-fill);
}
.prose.status-banner p { margin: 0 !important; }
.prose.status-banner p + p { margin-top: 6px !important; }

.block.compact-file { height: 90px !important; min-height: 0 !important; overflow: auto; }
.compact-file .empty { min-height: 0 !important; height: 100% !important; }

.main-action { min-height: 48px; font-size: 1rem !important; }
"""

## Les étapes affichées dans l'en-tête : (nom de l'onglet, ce qu'on y fait).
STEPS = (("1. Entraîner", "créer un modèle"), ("2. Suivre", "voir où il en est"),
         ("3. Tester", "le juger sur une image"), ("4. Valider", "le mesurer face à JPEG"),
         ("5. Optimisation Aidge", "le préparer pour la carte"))


def header():
    """! @brief Ajoute l'en-tête de la page : le but de la plateforme et les étapes."""
    steps = "".join(f"<li><b>{name}</b> : {what}</li>" for name, what in STEPS)
    gr.HTML('<div class="hero"><div class="eyebrow">AIDGE IN SPACE</div>'
            "<h1>Plateforme de compression d'images</h1>"
            "<p>Compresser les images d'un rover lunaire avec un réseau de neurones, pour économiser la bande passante. "
            "Entraînez un modèle, puis vérifiez qu'il bat les formats classiques.</p>"
            f"<ol>{steps}</ol></div>", padding=False, elem_classes="hero-block")


def intro(text, next_step=None):
    """! @brief Ajoute l'explication en tête d'un onglet.

    @param text       À quoi sert l'onglet (Markdown).
    @param next_step  Ce qui vient après (Markdown) ; None : rien de plus.
    """
    gr.Markdown(text + (f"\n\n**Ensuite :** {next_step}" if next_step else ""), elem_classes="tab-intro")


def hint(text):
    """! @brief Ajoute un court texte d'aide, discret (Markdown).

    @param text  Texte à afficher.
    """
    gr.Markdown(text, elem_classes="hint")


@contextmanager
def card(title, description=None):
    """! @brief Section encadrée : les composants créés dans le bloc @c with y sont placés.

    @param title        Titre de la section.
    @param description  Phrase sous le titre (Markdown) ; None : aucune.
    @return Gestionnaire de contexte qui fournit le gr.Column de la section.
    """
    with gr.Column(elem_classes="card") as column:
        gr.Markdown(f"### {title}" + (f"\n\n{description}" if description else ""), elem_classes="card-title")
        yield column
