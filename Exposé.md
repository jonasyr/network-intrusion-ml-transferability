Exposé
Internationale Hochschule Duales Studium
Studiengang: B. Sc. Informatik
Wie effektiv sind Machine-Learning-Modelle für Anomalieerkennung im
Netzwerkverkehr?
Eine experimentelle Analyse mit NSL-KDD und CIC-IDS-2017
Jonas Weirauch
Matrikelnummer: 10237021
Im Wiesengrund 19, 55286 Sulzheim
Betreuende Person: Dominic Lindner
Abgabedatum: 31.03.2025
II
Inhaltsverzeichnis
Inhaltsverzeichnis .......................................................................................................................... III
Tabellenverzeichnis........................................................................................................................ IV
Abkürzungsverzeichnis ................................................................................................................... V
1
 Einleitung ................................................................................................................................ 1
2
 Theoretische Fundierung ........................................................................................................ 3
3
 Methodik ................................................................................................................................. 5
4
 Geplante Gliederung der Projektarbeit .................................................................................... 6
5
 Zeitplan ................................................................................................................................... 7
Literaturverzeichnis ......................................................................................................................... 8
III
Tabellenverzeichnis
Tabelle 1:
 Gliederung Praxisprojekt.............................................................................................. 6
Tabelle 2:
 Zeitplan ........................................................................................................................ 7
IV
Abkürzungsverzeichnis
CIC
CIC-IDS-2017
DBSCAN
IDS
KI
KDD
LSTM
ML
NSL-KDD
PCA
SVM
Canadian Institute for Cybersecurity
Canadian Institute for Cybersecurity Intrusion Detection System 2017
(Datensatz)
Density-Based Spatial Clustering of Applications with Noise
Intrusion Detection System
Künstliche Intelligenz
Knowledge Discovery and Data Mining
Long Short-Term Memory
Machine Learning
Network Security Laboratory - Knowledge Discovery and Data Mining
(Datensatz, verbesserte Version von KDD Cup 99)
Principal Component Analysis
Support Vector Machines
V
1
 Einleitung
Netzwerke sind in der heutigen digitalen Welt ein wesentlicher Bestandteil moderner Organisationen
und Infrastrukturen. Die steigende Vernetzung und Digitalisierung führen dazu, dass die Gefahr von
Cyberangriffen stetig zunimmt. Gemäß dem Global Risk Report 2024 des Weltwirtschaftsforums
gehören Cyberangriffe zu den fünf bedeutendsten globalen Risiken in den nächsten Jahren (Global
Risks Report 2024, 2024). Die entstandenen finanziellen Verluste bis zum Jahr 2025 wurden auf
mehr als 10,5 Billionen US-Dollar jedes Jahr geschätzt, was einer Verdreifachung im Vergleich zu
2015 entspricht (Global Risks Report 2024, 2024; Taman, 2024). Die besorgniserregenden
Statistiken betonen die akute Erfordernis wirksamer Sicherheitsvorkehrungen zum Schutz wichtiger
Infrastruktureinrichtungen.
Eine wichtige Methode zur frühzeitigen Erkennung von Gefahren besteht darin, Anomalien im
Datenverkehr des Netzwerks zu identifizieren (Vinayakumar et al., 2019). Traditionelle Systeme, die
auf Signaturen basieren, erreichen zunehmend ihre Grenzen, da sie lediglich bekannte Muster
identifizieren können (Ring et al., 2019). Im Gegensatz dazu hat Machine Learning (ML) das
Potenzial,
 auch
 bisher
 unbekannte Angriffsmuster
 und
 Zero-Day-Exploits
 aufzudecken
(Vinayakumar et al., 2019).
In den vergangenen Jahren hat die Forschung im Bereich der Anomalieerkennung mittels
maschinellen Lernens erhebliche Fortschritte erzielt (Vinayakumar et al., 2019). Es werden
traditionelle Methoden wie Support Vector Machines (SVM) und Random Forests sowie moderne
Deep-Learning Modelle wie Long Short-Term Memory (LSTM) verwendet (Belavagi & Muniyal, 2016;
Vinayakumar et al., 2019; Zhou et al., 2020). Allerdings ist die tatsächliche Wirksamkeit dieser
Modelle in realen Netzwerken noch nicht vollständig geklärt (Ring et al., 2019). Die Netzwerkdaten
sind in vielerlei Hinsicht anspruchsvoll: Sie sind hochdimensional, kompliziert, veränderlich und
ungleichmäßig verteilt (Ring et al., 2019). Die Vielfalt von Netzwerkumgebungen macht es außerdem
schwierig, die Modelle zu verallgemeinern (Ring et al., 2019).
Ein zusätzliches Hindernis besteht darin, passende Datensätze zu finden. Häufig enthalten reale
Daten sensible Informationen und sind selten öffentlich zugänglich (Ring et al., 2019). Zwei
Datensätze, die weit verbreitet sind, sind NSL-KDD und CIC-IDS-2017 (IDS 2017 | Datasets |
Research | Canadian Institute for Cybersecurity | UNB, o. J.; NSL-KDD | Datasets | Research |
Canadian Institute for Cybersecurity | UNB, o. J.; Ring et al., 2019). Der erste ist eine optimierte
Variante des KDD-Cup-99 (Tavallaee et al., 2009), während der zweite realistischeren
Netzwerkverkehr mit aktuellen Angriffsszenarien bietet (Sharafaldin et al., 2018).
Es mangelt an systematischen, vergleichbaren Untersuchungen zur Beurteilung unterschiedlicher
ML-Modelle auf diesen Datensätzen, obwohl es viele Studien dazu gibt (Ring et al., 2019). Viele
Arbeiten fokussieren lediglich auf individuelle Algorithmen oder bestimmte Arten von Angriffen (Ring
1
et al., 2019). Selten werden Aspekte wie die Stabilität angesichts von Konzeptwechseln, die
Interpretierbarkeit von Modellen oder der Echtzeit-Ressourcenverbrauch in Betracht gezogen
(Gharib et al., 2016; Ring et al., 2019).
Diese
 Arbeit
 adressiert
 die
 Frage:
 „Wie
 effektiv
 sind
 Machine-Learning-Modelle
 für
Anomalieerkennung im Netzwerkverkehr?“ Ziel ist eine umfassende experimentelle Analyse mit
NSL-KDD und CIC-IDS-2017. Das Ziel besteht darin, eine ausführliche experimentelle
Untersuchung mit NSL-KDD und CIC-IDS-2017 durchzuführen. Zusätzlich zur Genauigkeit müssen
auch praktische Aspekte wie die Dauer des Trainings, die Zeit für die Inferenz und die Flexibilität in
Betracht gezogen werden. Die Resultate sollen konkrete Ratschläge für die effektive Anwendung
von KI-Modellen in verschiedenen Netzwerkszenarien bieten.
2
2
 Theoretische Fundierung
Die
 Erkennung von Anomalien im Netzwerkverkehr ist äußerst wichtig in modernen
Sicherheitsarchitekturen, um ungewöhnliche Muster zu erkennen, welche auf Angriffe wie Denial-of-
Service, unbefugtes Eindringen, Datenexfiltration oder Malware hindeuten könnten (Ring et al.,
2019). Dabei unterscheidet man grundsätzlich zwischen punktuellen, kontextuellen und kollektiven
Anomalien (Ring et al., 2019). Kollektive Anomalien beziehen sich auf Gruppen von Datenpunkten,
die gemeinsam ein unnatürliches Verhalten zeigen, obwohl einzelne Werte normal erscheinen
können (Ring et al., 2019).
Zur Erkennung werden zwei grundlegende Ansätze verfolgt: signaturbasierte und anomaliebasierte
Verfahren (Ring et al., 2019). Signaturbasierte Systeme vergleichen den aktuellen Verkehr mit
bekannten Angriffsmustern, was hohe Präzision bei bereits bekannten Angriffen möglich macht (Ring
et al., 2019). Es erkennt jedoch keine neuen, unbekannten Muster (Ring et al., 2019). Im Gegensatz
dazu modellieren anomaliebasierte Systeme dagegen zunächst das normale Netzwerkverhalten und
Kennzeichnen Abweichungen als potenziell gefährlich (Ring et al., 2019). Durch diesen Ansatz
können auch Zero-Day-Exploits und neuartige Angriffsmuster erfasst werden, was jedoch eine
präzise Modellierung des Normalverhaltens und die Festlegung geeigneter Schwellenwerte
voraussetzt, um Falsch-Positiv-Meldungen zu minimieren (Vinayakumar et al., 2019).
Für die praktische Umsetzung der Anomalieerkennung ist jedoch mehr nötig, als lediglich ein
geeignetes Modell auszuwählen. Um auf veränderte Netzwerkbedingungen und Concept Drift
reagieren zu können, ist es in realen IT-Umgebungen entscheidend, die Modelle kontinuierlich zu
überwachen und anzupassen (Ring et al., 2019). Auch die enge Kooperation zwischen
automatisierten Systemen und Fachleuten für Sicherheit ist hierbei von Bedeutung, da menschliche
Expertise oft notwendig ist, um Fehlalarme zu bestätigen und kritische Ereignisse richtig
einzuordnen. Dank dieser hybriden Methode lassen sich die Vorzüge der Automatisierung
ausschöpfen und zugleich flexibel auf komplizierte oder nicht vorhergesehene Begebenheiten
reagieren.
In diesem Zusammenhang haben sich Methoden des maschinellen Lernens als vielsprechende
Wekzeuge etabliert (Vinayakumar et al., 2019). Sie können in drei Kategorien eigeteilt werden:
überwachte, unüberwachte und Deep-Learning-Ansätze (Vinayakumar et al., 2019). Gelabelte
Daten werden von überwachten Verfahren wie Support Vector Machines, Entscheidungsbäumen,
Random Forests und k-nearest-Neighbours verwendet, um normales und anormales Verhalten zu
unterscheiden (Belavagi & Muniyal, 2016; Vinayakumar et al., 2019). Unüberwachte Methoden wie
Clustering mit k-means, dichtebasierte Verfahren wie DBSCAN oder Dimensionsreduktion mittels
Principal Comoponent Analysis benötigen keine vorherige Datenkennzeichnung und sind besonders
nützlich, wenn gelabelte Daten rar sind (Vinayakumar et al., 2019). Autoencoder, rekurrente
neuronale Netzwerke (insbesondere LSTM) und Convolutional Neural Networks sind Beispiele für
3
Deep-Learning-Techniken, die in der Lage sind, komplexe nichtlineare Beziehungen in
hochdimensionalen Daten zu erfassen (Aksu & Ali Aydin, 2018; Vinayakumar et al., 2019). Sie
benötigen jedoch große Mengen an Daten und Rechenressourcen (Vinayakumar et al., 2019).
Für die empirische Evaluation solcher Modelle haben sich zwei Datensätze als Referenz etabliert
(C. & M.P., 2022; Ring et al., 2019). Der NSL-KDD-Datensatz, eine verbesserte Version des KDD-
Cup-99 (Tavallaee et al., 2009), bietet ca. 125.973 Trainings- und 22.544 Testdatensätze mit 41
Merkmalen (NSL-KDD | Datasets | Research | Canadian Institute for Cybersecurity | UNB, o. J.) und
adressiert das Problem redundanter Daten (Tavallaee et al., 2009), spiegelt jedoch den simulierten
Netzwerkverkehr von 1998 wider (McHugh, 2000; Ring et al., 2019). Der CIC-IDS-2017-Datensatz
liefert aktuellere und realistischere Daten aus einer fünftätigen Netzwerkumgebung mit 25 Benutzern
(Sharafaldin et al., 2018), umfasst rund 2,8 Millionen Datenpunkte und 79 Merkmale (IDS 2017 |
Datasets | Research | Canadian Institute for Cybersecurity | UNB, o. J.), leidet aber unter
Datenungleichgewichten und einem fehlenden separaten Testdatensatz (Ring et al., 2019;
Sharafaldin et al., 2018).
Es sollte mehr Wert darauf gelegt werden, in künftigen Studien die Einbindung der entwickelten
Anomalieerkennungssysteme in bereits existierende Sicherheitsinfrastrukturen zu untersuchen. Es
ist wichtig, nicht nur die Genauigkeit der Erkennung und die Reaktionszeiten zu berücksichtigen,
sondern auch ökonomische Aspekte wie den Ressourcenverbrauch und die Anpassungsfähigkeit in
umfangreichen Netzwerken. Durch die Durchführung von Tests unter realen Bedingungen,
beispielsweise durch Pilotprojekte oder Simulationen, können wichtige Erkenntnisse darüber
gewonnen werden, wie sich theoretische Konzepte in der Praxis bewähren und welche Änderungen
erforderlich sind, um eine umfassende Sicherheitsüberwachung sicherzustellen.
Die Forschung hat bereits viele verschiedene Ansätze untersucht. Beispielsweise hat sich gezeigt,
dass der Random Forest Algorithmus im NSL-KDD-Datensatz häufig hervorragende Leistungen
erzielt (Mourouzis & Avgousti, 2021; Vinayakumar et al., 2019). Erste Untersuchungen mit CIC-IDS-
2017 zeigen ebenfalls die Vorzüge von kombinierten Methoden und einer gezielten Auswahl von
Merkmalen (Aksu & Ali Aydin, 2018; Zhou et al., 2020). Dennoch fehlen nach wie vor systematische
und vergleichende Untersuchungen, die beide Datensätze unter gleichen Bedingungen analysieren.
Häufig werden praxisrelevante Faktoren wie Trainings- und Inferenzzeiten, Ressourcenverbrauch
und die Fähigkeit der Modelle, sich an sich ändernde Netzwerkbedingungen anzupassen, nicht
ausreichend berücksichtigt (Gharib et al., 2016; Ring et al., 2019). Das Ziel dieser Arbeit besteht
darin, die bestehenden Lücken zu füllen und ein tiefergehendes Verständnis dafür zu entwickeln,
wie effektiv verschiedene Machine-Learning-Modelle zur Erkennung von Netzwerkanomalien sind.
4
3
 Methodik
Die geplante experimentelle Analyse ist darauf ausgerichtet, in einem realistischen Kontext
durchgeführt zu werden, der eine harmonische Balance zwischen Anstrengung, Komplexität und
signigikanten Resultaten sicherstellt. Das Ziel besteht darin, die Wirksamkeit spezifischer Machine-
Learning-Modelle, wie Random Forest, Support Vector Machine und Autoencoder, für die
Anomalieerkennung im Netzwerkverkehr zu untersuchen. Die Methodik basiert auf den zuvor
präsentierten theoretischen Grundlagen und fokussiert sich auf Strategien, die sowohl etablierte
Angriffsmuster als auch mögliche neue Bedrohungen identifizieren können, ohne den Rahmen der
Untersuchung überflüssig zu erweitern.
Zunächst unterliegen die existierenden Datensätze NSL-KDD und CIC-IDS-2017 einer normalen
Vorverarbeitung. Dazu gehört das Beheben von Unstimmigkeiten und Fehlwerten, die Angleichung
numerischer Merkmale und die Umwandlung kategorialer Daten durch übliche Kodierungsverfahren.
Des Weiteren ist beabsichtigt durch eine präzise Feature-Auswahl die Dimensionalität zu verringern
und die Effektivität des Trainings zu steigern. Zur Minimierung der Verzerrung von Ergebnissen und
zur Steigerung der Aussagekraft von Modellen kommen Maßnahmen zur Behandlung des
Klassenungleichgewichts, beispielsweise Oversampling, zur Anwendung.
Die Implementierung der Modelle findet in Python statt, wobei bewährte Bibliotheken verwendet
werden.
 Eine
 pragmatische
 Hyperparameter-Optimierung
 wird
 eingesetzt,
 um
 durch
Kreuzvalidierung die optimalen Modelleinstellungen zu identifizieren. Um vergleichbare Resultate zu
erzielen, werden sämtliche Versuche unter konsistenten Bedingungen durchgeführt. Zusätzlich zu
den traditionellen Klassifikationsmetriken wie Genauigkeit, Präzision, Rückruf und F1-Score werden
auch Effizienzindikatoren wie die Dauer des Trainingsprozesses und der Inferrenzzeit berücksichtigt.
Durch Cross-Dataset-Experimente erfolgt zudem eine Überprüfung der Generalisierungsfähigkeit
der Modelle, um zu gewährleisten, dass die erarbeiteten Resultate ebenfalls auf realistische
Netzwerkszenarien anwendbar sind.
Die Anwendung dieser methodischen Strategie gestattet eine gründliche Untersuchung der
ausgewählten Modelle, ohne den Umfang der Untersuchung unnötig zu erweitern. Sie gewährleistet,
dass
 sowohl
 die
 Präzision
 der
 Erkennung
 als
 auch
 die
 praktische Anwendung
 in
sicherheitsrelevanten Kontexten angemessen berücksichtigt werden. Infolgedessen trägt die
Untersuchung bedeutend zur gegenwärtigen Forschungslandschaft bei und bietet praxisorientierte
Ratschläge für die Anwendung von Machine-Learning-Modellen in der Identifizierung von
Netzwerkanomalien.
5
4
 Geplante Gliederung der Projektarbeit
Für die im 4.Tabelle 1:
1
1.1
1.2
1.3
2
2.1
2.2
2.3
3
3.1
3.2
3.3
4
4.1
4.2
4.3
5
5.1
5.2
5.3
Semester anzufertigende Projektarbeit ist folgende GliederungsstrukturGliederung Praxisprojekt
Einleitung
Hinführung zum Thema
Zentrale Begrifflichkeiten
Ziel und Struktur der Arbeit
Theoretische Fundierung
Grundlagen der Netzwerkanomalieerkennung
Machine-Learning-Ansätze zur Intrusion Detection
Relevante Darensätze: NSL-KDD und CIC-IDS-2017
Methodik
Datensatzbeschreibung und Vorverarbeitung
Ausgewählte
 Machine-Learning-Modelle
 und
Implementierung
Evaluationsmetriken und experimentelles Design
Ergebnisdarstellung und -interpretation
Ergebnisse der Modellbewertung auf dem NSL-KDD-
Datensatz
Ergebnisse der Modellbewertung auf dem CIC-IDS-2017-
Datensatz
Vergleichende Analyse, Effizienz und Generalisierbarkeit
Fazit
Zusammenfassung
Implikationen für die Praxis
Limitationen und Implikationen für die Wissenschaft
Summe
vorgesehen:
ca. 1,5 Seiten
ca. 0,5 Seiten
ca. 0,5 Seiten
ca. 0,5 Seiten
ca. 4 Seiten
ca. 1 Seiten
ca. 1,5 Seiten
ca. 1,5 Seiten
ca. 3 Seiten
ca. 1 Seiten
ca. 1 Seiten
ca. 1 Seiten
ca. 5 Seiten
ca. 1,5 Seiten
ca. 1,5 Seiten
ca. 2 Seiten
ca. 2 Seiten
ca. 0,75 Seiten
ca. 0,75 Seiten
ca. 0,5 Seiten
ca. 15 Seiten
Quelle:
 Eigene Darstellung
Durch diese Struktur wird eine deutliche Abgrenzung zwischen den theoretischen Grundlagen, dem
methodischen Vorgehen und der Analyse der Ergebnisse sichergestellt. Die geplanten Seitenzahlen
reflektieren die Schwerpunkte der Studie, wobei der empirischen Analyse und der Auslegung der
Ergebnisse besondere Bedeutung beigemessen wird, um das Forschungsziel vollständig zu
erreichen, nämlich die Bewertung der Wirksamkeit von ML-Modellen zur Erkennung von
Netzwerkanomalien.
6
5
 Zeitplan
Unter Berücksichtigung der sonstigen Vorlesungen, Verpflichtungen und Nichtverfügbarkeiten im
Sommersemester 2025 ist folgende Projektplanung vorgesehen.
Tabelle 2:
 Zeitplan
Phase/Arbeitspaket
1. Kick-off, Einarbeitung
& Einleitung
2. Theoretische
Fundierung
3. Methodik & Setup
Zeitraum
KW 14-15
KW 15-18
KW 18 – KW 20
4. Datensatzvorbereitung
& Experimente
5. Ergebnissaufbereitung
& Interpretation
6. Fazit & Implikationen
KW 20 – KW 24
KW 23 – KW 26
KW 26 – KW 27
7. Puffer, Überarbeitung
& Abgabe
Gesamt
KW 27 – KW 30
KW 14 – KW 30
Dauer (ca.)
2 Wochen
4 Wochen
3 Wochen
5 Wochen
4 Wochen
2 Wochen
4 Wochen
17 Wochen
Fokus/Meilensteine
Feinplanung, erste Literatur,
Rohfassung Einleitung (Kapitel 1)
Vertiefte Literaturrecherche,
Schreiben Kapitel 2
Detailplanung Experimente, Code-
Implementierung, Schreiben
Kapitel 3
Datenaufbereitung (NSL-KDD,
CIC-IDS), Modelldurchläufe
Auswertung Rohdaten, Grafiken
erstellen, Schreiben Kapitel 4
Schlussfolgerungen ziehen,
Schreiben Kapitel 5
Gesamtdurchsicht, Korrekturen,
Formatierung, Finale Abgabe
Vorraussichtliche Abgabe:
Mitte/Ende Juli 2025
Quelle:
 Eigene Darstellung
Dieser Zeitplan dient als Richtlinie und erfordert eine kontinuierliche Selbstorganisation, um die
Meilensteine trotz der begrenzten Zeitfenster unter der Woche termingerecht zu erreichen. Die
Pufferzeit am Ende ist für unvorhergesehene Verzögerungen und eine gründliche finale
Überarbeitung vorgesehen.
7
Literaturverzeichnis
Aksu, D., & Ali Aydin, M. (2018). Detecting Port Scan Attempts with Comparative Analysis of Deep
Learning and Support Vector Machine Algorithms. 2018 International Congress on Big Data,
Deep
 Learning
 and
 Fighting
 Cyber
 Terrorism
 (IBIGDELFT),
 77–80.
https://doi.org/10.1109/IBIGDELFT.2018.8625370
Belavagi, M. C., & Muniyal, B. (2016). Performance Evaluation of Supervised Machine Learning
Algorithms
 for
 Intrusion
 Detection.
 Procedia
 Computer
 Science,
 89,
 117–123.
https://doi.org/10.1016/j.procs.2016.06.016
C., H., & M.P., P. J. (2022). A Review of Benchmark Datasets and its Impact on Network Intrusion
Detection Techniques. 2022 Fourth International Conference on Cognitive Computing and
Information Processing (CCIP), 1–6. https://doi.org/10.1109/CCIP57447.2022.10058660
Gharib, A., Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2016). An Evaluation Framework for
Intrusion Detection Dataset. 2016 International Conference on Information Science and
Security (ICISS), 1–6. https://doi.org/10.1109/ICISSEC.2016.7885840
Global
 Risks
 Report
 2024.
 (2024).
 World
 Economic
 Forum.
https://www.weforum.org/publications/global-risks-report-2024/
IDS 2017 | Datasets | Research | Canadian Institute for Cybersecurity | UNB. (o. J.). Abgerufen 29.
März 2025, von https://www.unb.ca/cic/datasets/ids-2017.html
McHugh, J. (2000). Testing Intrusion detection systems: A critique of the 1998 and 1999 DARPA
intrusion detection system evaluations as performed by Lincoln Laboratory. ACM
Transactions
 on
 Information
 and
 System
 Security,
 3(4),
 262–294.
https://doi.org/10.1145/382912.382923
Mourouzis, T., & Avgousti, A. (2021). Intrusion Detection with Machine Learning Using Open-Sourced
Datasets (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2107.12621
Moustafa, N., & Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical
analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set.
Information
 Security
 Journal:
 A
 Global
 Perspective,
 25(1–3),
 18–31.
https://doi.org/10.1080/19393555.2015.1125974
NSL-KDD | Datasets | Research | Canadian Institute for Cybersecurity | UNB. (o. J.). Abgerufen 29.
März 2025, von https://www.unb.ca/cic/datasets/nsl.html
8
Ring, M., Wunderlich, S., Scheuring, D., Landes, D., & Hotho, A. (2019). A Survey of Network-based
Intrusion
 Detection
 Data
 Sets.
 Computers
 &
 Security,
 86,
 147–167.
https://doi.org/10.1016/j.cose.2019.06.005
Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion
Detection Dataset and Intrusion Traffic Characterization: Proceedings of the 4th
International Conference on Information Systems Security and Privacy, 108–116.
https://doi.org/10.5220/0006639801080116
Taman, D. (2024). Impacts of Financial Cybercrime on Institutions and Companies. بادلآل ةيبرعلا ةلجملا
488–477 ,(30)8 ,ةيناسنلإا تاساردلاو. https://doi.org/10.21608/ajahs.2024.341707
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99
data set. 2009 IEEE Symposium on Computational Intelligence for Security and Defense
Applications, 1–6. https://doi.org/10.1109/CISDA.2009.5356528
Vinayakumar, R., Alazab, M., Soman, K. P., Poornachandran, P., Al-Nemrat, A., & Venkatraman, S.
(2019). Deep Learning Approach for Intelligent Intrusion Detection System. IEEE Access,
7, 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334
Zhou, Y., Cheng, G., Jiang, S., & Dai, M. (2020). Building an efficient intrusion detection system
based on feature selection and ensemble classifier. Computer Networks, 174, 107247.
https://doi.org/10.1016/j.comnet.2020.107247