# Task2 Agent

To jest agent do `task2`, ktory pracuje na lokalnych repozytoriach z `data/repositories-{lang}-{stage}`.
Nie pobiera plikow z internetu i nie wysyla nic do verify. Jego zadanie to dla kazdego datapointu
znalezc przydatny kontekst i zapisac go do predictions JSONL.

## Langfuse

Jesli chcesz sledzic eksperymenty w Langfuse, dodaj do glownego `.env`:

```env
LANGFUSE_PUBLIC_KEY="..."
LANGFUSE_SECRET_KEY="..."
LANGFUSE_HOST="https://cloud.langfuse.com"
TASK2_EXPERIMENT_NAME="baseline-1"
```

`LANGFUSE_HOST` jest opcjonalny, jesli korzystasz z domyslnego hosta albo self-hosted zgodnego z ustawieniami SDK.
Jesli chcesz tymczasowo wylaczyc eksport trace'y bez ruszania kluczy, ustaw `TASK2_DISABLE_LANGFUSE=1`.

Po wlaczeniu Langfuse:

- `agent.app` zapisuje trace dla kazdego datapointu agenta,
- `agent.complete_and_eval` zapisuje trace dla kazdego datapointu benchmarku completion,
- kazde wywolanie `OpenAI responses.create` loguje sie automatycznie przez SDK Langfuse, razem z usage i request/response,
- kroki tooli sa zapisywane jako osobne spany podrzedne,
- benchmark zapisuje tez score'y proxy, np. `completion_reference_chrf_proxy`,
- artefakty agenta zawieraja `trace.json` z `trace_id` i `trace_url`.

Nie trzeba ustawiac dodatkowego przelacznika dla auto-instrumentacji OpenAI. Jesli sa klucze `LANGFUSE_*`
w `.env`, logowanie jest wlaczone automatycznie. Zeby je wylaczyc, ustaw tylko `TASK2_DISABLE_LANGFUSE=1`.

## Precision Context

Ten agent nie sklada juz `context` jako dowolnego stringa. Zamiast tego:

- eksploruje repo przez `inspect_target`, `list_files`, `read_file`, `search_files`,
- dodaje tylko wybrane fragmenty przez `add_context_snippet`,
- sprawdza budzet i podglad przez `preview_context`,
- moze wyczyscic wybor przez `reset_context`,
- finalizuje context przez `finish`, ktory sklada go automatycznie ze snippetow.

Domyslne ograniczenia:

- `TASK2_CONTEXT_CHAR_BUDGET=16000`
- `TASK2_CONTEXT_MAX_SNIPPETS=8`

## Narzedzia

Agent ma do dyspozycji:

- `inspect_target`
  Zwraca `path`, `modified`, koniec `prefix`, poczatek `suffix` i zawartosc katalogu docelowego.
- `list_files`
  Listuje katalogi i pliki w repo.
- `read_file`
  Czyta dokladne zakresy linii z pliku.
- `search_files`
  Szuka fraz i symboli w plikach tekstowych.
- `add_context_snippet`
  Dodaje dokladny snippet do zarzadzanego contextu, jesli miesci sie w budzecie.
- `preview_context`
  Pokazuje aktualnie wybrane snippety i pozostaly budzet.
- `reset_context`
  Czyści aktualny zestaw snippetow.
- `finish`
  Zwraca gotowy `context` z aktualnie wybranych snippetow.

## Uruchomienie

Najpierw przygotuj dane, jesli jeszcze nie sa rozpakowane:

```bash
cd /home/mikolaj/work/tasks-2026/task2/EnsembleAI2026-starter-kit
bash prepare_data.sh start python
```

Potem uruchom generator predictions:

```bash
/home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.app \
  --stage start \
  --lang python \
  --limit 1
```

Domyslny output:

```text
predictions/python-start-agent.jsonl
```

Przykladowe uruchomienie dla wiekszego zbioru:

```bash
/home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.app \
  --stage practice \
  --lang python \
  --limit 10
```

Przy dluzszych przebiegach mozesz przyspieszyc run przez rownolegle przetwarzanie datapointow:

```bash
TASK2_MAX_AGENT_STEPS=16 /home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.app \
  --stage public \
  --lang python \
  --workers 4 \
  --output predictions/python-public-agent.jsonl
```

Jesli chcesz miec wiecej workerow datapointow, ale ograniczyc realna liczbe rownoleglych requestow do OpenAI, ustaw:

```bash
TASK2_OPENAI_MAX_CONCURRENT_REQUESTS=4
```

To jest przydatne przy `429` na TPM, bo pozwala trzymac `--workers` wyzej bez zalewania API naraz.

Jesli run zostanie przerwany, mozesz go wznowic bez utraty juz zapisanych wynikow:

```bash
TASK2_MAX_AGENT_STEPS=16 /home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.app \
  --stage public \
  --lang python \
  --workers 4 \
  --resume \
  --output predictions/python-public-agent.jsonl
```

## Lokalna ewaluacja

Mozesz uruchomic lokalny evaluator proxy:

```bash
/home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.evaluate \
  --stage start \
  --lang python \
  --predictions-file predictions/python-start-agent.jsonl
```

To nie jest oficjalny score konkursowy. Skrypt:

- probuje odzyskac referencje z lokalnego repo,
- sprawdza, czy `context` zawiera `target_path`,
- sprawdza, czy `context` zawiera odzyskany reference,
- liczy proxy `chrF` miedzy `context` i reference tam, gdzie reference da sie odzyskac.

Oficjalnej oceny 1:1 lokalnie nie odtworzysz bez modeli i pipeline organizatora.

## Lokalny benchmark end-to-end

Mozesz tez odpalic drugi krok lokalnie:

- bierzesz gotowy `context` z `predictions/*.jsonl`,
- uruchamiasz lokalny model completion na `context + prefix + suffix`,
- porownujesz completion z odzyskanym reference tam, gdzie reference da sie odzyskac.

Przyklad:

```bash
/home/mikolaj/work/tasks-2026/.venv/bin/python -m agent.complete_and_eval \
  --stage practice \
  --lang python \
  --predictions-file predictions/python-practice-agent.jsonl \
  --limit 10 \
  --only-recovered
```

Domyslnie szczegolowe wyniki sa zapisywane do:

```text
agent/workspace/completion_evals/{lang}-{stage}.jsonl
```

Wymaga to ustawionego `OPENAI_API_KEY` albo `OPENAI_API_TOKEN`, bo ten krok wykonuje lokalne completion modelem.
To nadal jest benchmark proxy, a nie oficjalna ewaluacja konkursowa.
Kazdy wiersz wyniku zawiera tez `langfuse_trace_id` i `langfuse_trace_url`, jesli Langfuse jest skonfigurowany.
