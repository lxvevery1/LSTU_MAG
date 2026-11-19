# Сборка отчёта: make            (два прохода xelatex — для оглавления)
# Нужны xelatex и шрифты Liberation Serif/Sans/Mono.
# Таблицы генерируются из результатов: ../src/make_tables.py

TEX = report.tex
AUX = $(TEX:.tex=.aux) $(TEX:.tex=.log) $(TEX:.tex=.out) \
      $(TEX:.tex=.toc) $(TEX:.tex=.pdf)

all: report.pdf

report.pdf: $(TEX) tables/*.tex ../results/figs/*.png
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)

tables:
	cd .. && .venv/bin/python src/make_tables.py

clean:
	rm -f $(AUX)

.PHONY: all tables clean