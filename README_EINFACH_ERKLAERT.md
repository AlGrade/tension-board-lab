# Das Modell einfach erklärt

Ganz kurz gesagt:

> Du gibst dem Modell einen Boulder mit seinen verwendeten Griffen und der
> Wandneigung. Das Modell antwortet mit einer V-Grade und einer Confidence.

## Was ist der Input?

Für einen Boulder bekommt das Modell folgende Informationen.

### 1. Das Layout

Das Modell erfährt, ob der Boulder auf dem **Mirror Layout** oder dem **Spray
Layout** geschraubt ist.

### 2. Die Wandneigung

Zum Beispiel:

```text
40°
```

Der Winkel wird als normale Zahl verarbeitet. Er ist keine Kategorie wie
„Winkel Nummer 2“.

### 3. Alle verwendeten Griffe

Für jeden verwendeten Griff kennt das Modell:

- den Grifftyp, zum Beispiel den Tension-Hold `20D`;
- das Material: Holz oder Plastik;
- die Variante, zum Beispiel links oder rechts;
- die Drehung des Griffs, zum Beispiel `45°`;
- die Position auf dem Board als X- und Y-Koordinate;
- die Verwendung im Boulder: Start, Hand, Fuß oder Finish.

Die Abstände zwischen den Griffen müssen nicht extra eingegeben werden. Das Modell
berechnet sie automatisch aus den Koordinaten.

Es berechnet unter anderem:

- den horizontalen Abstand;
- den vertikalen Abstand;
- den direkten Abstand;
- die Bewegungsrichtung;
- wie sich dieser Abstand bei der aktuellen Wandneigung auswirkt.

Die Aurora-Placement-ID bleibt zwar zur Nachverfolgung in den Daten, wird aber
**nicht als Input an das Modell übergeben**.

Das Modell soll also ungefähr Folgendes lernen:

> Dieser Grifftyp, so gedreht und in diesem Abstand zu den anderen Griffen, ist
> wahrscheinlich schwierig.

Es soll nicht einfach lernen:

> Placement 433 bedeutet meistens V8.

## Was ist der Output?

Eine Vorhersage sieht beispielsweise so aus:

```json
{
  "predicted_grade": "V9",
  "confidence": 0.2808
}
```

- `predicted_grade` ist die wahrscheinlichste V-Grade.
- `confidence` zeigt, wie stark sich das Modell für genau diese Grade entscheidet.

Eine Confidence von `0.28` bedeutet nicht, dass der Boulder mit 28-prozentiger
Sicherheit geklettert wird. Es bedeutet nur, dass das Modell dieser Grade 28 %
seiner gesamten Wahrscheinlichkeit gibt. Der Rest verteilt sich hauptsächlich auf
benachbarte Grades.

## Worauf wurde das Modell trainiert?

Verwendet wurden **21.809 Boulder-Winkel-Kombinationen** aus der bestehenden lokalen
Tension-Datenbank:

- 13.043 Mirror-Beispiele;
- 8.766 Spray-Beispiele;
- Winkel von 35°, 40°, 45°, 50° und 55°;
- Grades von V0 bis V14.

Ein Boulder bei 40° und derselbe Boulder bei 45° sind zwei Trainingsbeispiele. Der
Boulder kann bei den beiden Neigungen unterschiedliche Community-Grades haben.

Das Trainingsziel war immer:

> Welche Grade hat die Community diesem Boulder bei genau dieser Neigung gegeben?

Ein einzelnes Trainingsbeispiel sieht gedanklich ungefähr so aus:

```text
Layout: Mirror
Winkel: 40°
Griffe: A, B, C, D ...
Community-Grade: V8
```

Das Modell macht zuerst eine Vorhersage. Danach vergleicht es seine Vorhersage mit
der tatsächlichen Community-Grade `V8` und verändert seine internen Parameter ein
kleines Stück. Das wird mit Tausenden Bouldern immer wieder gemacht.

Die Anzahl der Ascents wird nur verwendet, um Community-Grades mit mehr
Wiederholungen beim Training etwas stärker zu gewichten. Die Anzahl der Ascents ist
später kein Input für eine Vorhersage.

## Wie werden Mirror und Spray behandelt?

Es gibt **ein gemeinsames Modell** für beide Layouts.

Das Modell wird nicht einmal für Mirror und einmal für Spray trainiert. Stattdessen
sieht es gemischte Beispiele:

```text
Mirror, 40°, Griffe ..., V8
Spray, 45°, Griffe ..., V7
Mirror, 35°, Griffe ..., V6
```

Dadurch kann es allgemeine Zusammenhänge gemeinsam lernen:

- bestimmte Grifftypen;
- schlechte oder gute Griffausrichtungen;
- weite Züge;
- starke Höhenunterschiede;
- Kombinationen aus Holz und Plastik.

Zusätzlich bekommt es immer die Information `mirror` oder `spray`. Dadurch kann es
lernen, dass sich die beiden Layouts eventuell systematisch unterschiedlich
verhalten.

Bei Mirror gibt es außerdem eine Schutzmaßnahme für den Test: Wenn ein Boulder nur
gespiegelt oder unter einem anderen Namen gespeichert wurde, bleiben diese Versionen
im selben Datensplit. Es kann daher nicht eine Version im Training und die
gespiegelte Kopie im Test landen.

## Wie werden die Neigungen behandelt?

Der Winkel wird als kontinuierliche Zahl eingegeben:

```text
35.0
40.0
45.0
50.0
55.0
```

Das Modell lernt also nicht nur:

```text
40° = Winkel-Kategorie 2
```

Stattdessen kann es Zusammenhänge dieser Art lernen:

```text
45° ist steiler als 40°.
Der gleiche vertikale Griffabstand wirkt bei 45° anders als bei 35°.
```

Der Winkel fließt auch direkt in die Berechnung der Bewegungsgeometrie ein.

Momentan wurde das Modell nur mit 35°, 40°, 45°, 50° und 55° trainiert. Technisch
könnte man beispielsweise 42° eingeben. Einer solchen Vorhersage sollten wir aber
noch nicht stark vertrauen, weil das Modell keine echten 42°-Trainingsbeispiele
gesehen hat.

## Wie wurden Training und Test getrennt?

Die Daten wurden in drei Teile aufgeteilt:

- **17.443 Beispiele für das Training**;
- **2.166 Beispiele für die Validierung**;
- **2.200 Beispiele für den abschließenden Test**.

Mit den Trainingsdaten lernt das Modell. Die Validierungsdaten helfen dabei zu
entscheiden, wann das Training gestoppt werden sollte. Die Testdaten werden erst am
Ende für die endgültige Bewertung verwendet.

Das Modell durfte die Testboulder während des Trainings nicht sehen. Identische,
umbenannte oder gespiegelte Konfigurationen werden gemeinsam demselben Split
zugeordnet. Dadurch soll verhindert werden, dass praktisch derselbe Boulder in
Training und Test auftaucht.

## Wie gut ist das aktuelle Modell?

Auf dem unangetasteten Testset erreicht es:

- einen durchschnittlichen Fehler von ungefähr **1,05 V-Grades**;
- 73,7 % der Vorhersagen liegen höchstens eine V-Grade daneben;
- 30,7 % der Vorhersagen treffen die Grade exakt.

Sehr kurz zusammengefasst:

```text
Grifftypen + Drehungen + Positionen + Layout + Winkel
                         ↓
               Graph Transformer
                         ↓
               V-Grade + Confidence
```
