# Das Modell einfach erklärt

Ganz kurz gesagt:

> Du gibst dem Modell die Neigung und die verwendeten Griffe. Es antwortet mit einer
> V-Grade und einer Confidence.

## Input

Das Modell bekommt die Wandneigung, zum Beispiel `40°`.

Für jeden verwendeten Griff bekommt es nur:

- den Grifftyp;
- die Drehung;
- die X/Y-Position;
- Start, Hand, Fuß oder Finish.

Aus den Positionen berechnet es automatisch Abstände zwischen allen Griffen. Die
Neigung fließt ebenfalls in diese Geometrie ein.

Nicht eingegeben werden Mirror oder Spray, separates Material, Links/Rechts-Variante,
Placement-ID, Bouldername, Ascents oder die echte Grade.

Mirror und Spray werden gemeinsam zum Trainieren verwendet. Das Modell erfährt aber
nicht, von welchem Layout ein Beispiel stammt. Der Grifftyp bleibt notwendig: Ohne
ihn könnte es gute und schlechte Griffe an derselben Position nicht unterscheiden.

## Output

```json
{
  "predicted_grade": "V8",
  "confidence": 0.5128
}
```

`predicted_grade` ist die wahrscheinlichste Grade. `confidence` zeigt, wie viel der
internen Wahrscheinlichkeit das Modell dieser Grade gibt. Sie ist keine Garantie.

## Training

Das Modell wurde mit 21.809 Boulder-Winkel-Beispielen trainiert:

- 13.043 Mirror-Beispiele;
- 8.766 Spray-Beispiele;
- Winkel 35°, 40°, 45°, 50° und 55°;
- Community-Grades V0 bis V14.

Ein Boulder bei 40° und bei 45° sind zwei Beispiele. Ascents gewichten nur die
Zuverlässigkeit einer Trainingsgrade; sie werden nicht an das Modell übergeben.

Die Daten sind aufgeteilt in 17.477 Training, 2.158 Validierung und 2.174 Test.
Identische, umbenannte oder gespiegelte Konfigurationen bleiben im selben Teil. Es
gibt keine gemeinsame Input-Konfiguration zwischen Training und Test.

## Ergebnis

Auf dem unangetasteten Testset erreicht das Modell:

- durchschnittlicher Fehler: **1,028 V-Grades**;
- höchstens eine Grade daneben: **73,64 %**;
- exakt getroffen: **32,84 %**.

Es hat ungefähr 2,85 Millionen Parameter, sechs Transformer-Schichten und acht
Attention-Heads.

```text
Grifftyp + Drehung + Position + Rolle + Neigung
                       ↓
              Graph Transformer
                       ↓
              V-Grade + Confidence
```
