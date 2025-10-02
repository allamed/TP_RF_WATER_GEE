# TP IA Ressources en eau — Eau / Non‑eau (Sentinel‑2 + JRC GSW)

**Durée indicative :** 90–120 min
**Niveau :** Débutant à intermédiaire (GEE + ML)
**Données :** Sentinel‑2 SR harmonisée, JRC Global Surface Water (GSW), SRTM (pente)

---

## 🎯 Objectif du TP

Construire un **classificateur supervisé** (Random Forest) qui sépare **Eau / Non‑eau** à partir d’images Sentinel‑2, en s’appuyant sur des **étiquettes automatiques** issues du produit **JRC Global Surface Water (GSW)**, puis :

* générer une **carte binaire Eau/Non‑eau** sur une zone d’intérêt (AOI),
* évaluer le modèle (matrice de confusion, accuracy, kappa),
* produire une **série temporelle mensuelle** de la surface en eau (en hectares),
* exporter la carte et la série temporelle.

**Image mentale :** on demande à GSW de nous indiquer les zones presque certainement "eau" et "non‑eau" ⇒ on en prélève des points d’entraînement ⇒ on apprend à un modèle à **reconnaître** l’eau dans Sentinel‑2 ⇒ on applique ce modèle à d’autres mois pour **compter** la surface d’eau.

---

## 🧪 Comment appliquer ce TP

1. Ouvrir **Google Earth Engine Code Editor** (code.earthengine.google.com).
2. **Créer un nouveau script** et **coller progressivement** les blocs de code ci‑dessous **dans l’ordre**.
3. Après chaque étape, cliquer sur **Run** et **observer** les sorties (carte, console, graphiques).
4. Quand c’est proposé, activer les **Exports** pour récupérer les résultats dans Google Drive (dossier `GEE_Exports`).

> 🔁 Les blocs sont **cumulatifs** : collez chaque étape **à la suite** de la précédente.

---

## Étape 0 — Paramètres globaux et options

**Ce que nous faisons (explication simple)**
On définit :

* la **zone d’étude** (AOI),
* les **périodes** d’entraînement et d’analyse,
* les **seuils** GSW pour dire « eau quasi‑certaine » / « non‑eau quasi‑certaine »,
* la **taille d’échantillonnage** pour le ML,
* des **commutateurs d’export**.

**Code à coller**

```javascript
/**** TP IA Ressources en eau — Eau / Non-eau (Sentinel-2 + GSW) ****/
/**** Auteur: Vous | Durée: ~90-120 min ****/

/* ===================== 0) PARAMS GLOBAUX ===================== */
var AOI = ee.Geometry.Rectangle([-6.55, 31.98, -6.29, 32.18]).simplify(100);
var TRAIN_START = '2022-06-01';
var TRAIN_END   = '2022-12-31';
var TIMESERIES_START = '2022-01-01';
var TIMESERIES_END   = '2025-09-01';

// Auto-étiquetage (JRC GSW occurrence)
var GSW_WATER_MIN    = 80; // ≥ eau quasi-certaine
var GSW_NONWATER_MAX = 5;  // ≤ non-eau quasi-certaine

// Échantillonnage stratifié
var N_POINTS_TOTAL = 4000; // 2000/2000 par classe
var SEED = 42;

// Exports (Drive)
var DO_EXPORT_MAP = true;   // passer à true pour exporter la carte RF
var DO_EXPORT_TS  = true;   // passer à true pour exporter la série temporelle CSV
```

**Ce que vous devez voir**
Rien sur la carte pour l’instant ; juste la préparation des paramètres.

---

## Étape 1 — Outils utiles (masque nuages, indices, Otsu)

**Ce que nous faisons**

* Un **masque nuages** pour Sentinel‑2 (on retire nuages/cirrus).
* Des **indices spectro** (NDWI, MNDWI, NDVI) qui aident à distinguer l’eau.
* Une petite fonction **Otsu** pour trouver automatiquement un seuil sur NDWI (option pour la série temporelle).

**Analogie :** le masque nuages = on enlève les pièces « abîmées » du puzzle; les indices = des **filtres** qui rendent l’eau plus visible.

**Code à coller (à la suite)**

```javascript
/* ===================== 1) OUTILS ===================== */
// Masque nuages Sentinel-2 SR harmonisée (QA60)
function maskS2sr(img) {
  var qa = img.select('QA60');
  var cloudBitMask = 1 << 10;   // clouds
  var cirrusBitMask = 1 << 11;  // cirrus
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return img.updateMask(mask).copyProperties(img, img.propertyNames());
}

// Ajout d’indices
function addIndices(img) {
  var green = img.select('B3');
  var red   = img.select('B4');
  var nir   = img.select('B8');
  var swir1 = img.select('B11');
  var ndwi  = green.subtract(nir).divide(green.add(nir)).rename('NDWI');
  var mndwi = green.subtract(swir1).divide(green.add(swir1)).rename('MNDWI');
  var ndvi  = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
  return img.addBands([ndwi, mndwi, ndvi]);
}

// Petit Otsu (pour l’option NDWI dans la série temporelle)
function otsu(histDict) {
  var counts = ee.Array(ee.Dictionary(histDict).get('histogram'));
  var bins = ee.Array(ee.Dictionary(histDict).get('bucketMeans'));
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = bins.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = ee.Number(sum).divide(total);

  var bVarMax = ee.Number(0);
  var threshold = ee.Number(0);

  var countB = ee.Number(0);
  var sumB = ee.Number(0);

  var n = counts.length().get([0]);

  var iterate = ee.List.sequence(0, ee.Number(n).subtract(1)).iterate(function(idx, prev) {
    prev = ee.Dictionary(prev);
    var cB = ee.Number(prev.get('countB'));
    var sB = ee.Number(prev.get('sumB'));
    var maxVar = ee.Number(prev.get('bVarMax'));
    var thr = ee.Number(prev.get('threshold'));

    var c = counts.get([idx]);
    var b = bins.get([idx]);

    var cB2 = cB.add(c);
    var sB2 = sB.add(ee.Number(b).multiply(c));

    var wB = cB2.divide(total);
    var wF = ee.Number(1.0).subtract(wB);

    var mB = sB2.divide(cB2);
    var mF = ee.Number(sum).subtract(sB2).divide(total.subtract(cB2));

    var bVar = wB.multiply(wF).multiply(ee.Number(mB).subtract(mF).pow(2));

    var cond = bVar.gt(maxVar);
    return ee.Dictionary({
      countB: cB2,
      sumB: sB2,
      bVarMax: ee.Number(ee.Algorithms.If(cond, bVar, maxVar)),
      threshold: ee.Number(ee.Algorithms.If(cond, b, thr))
    });
  }, {countB: countB, sumB: sumB, bVarMax: bVarMax, threshold: threshold});

  return ee.Dictionary(iterate).get('threshold');
}
```

**Ce que vous devez voir**
Toujours rien de visible : ce sont des fonctions utilitaires utilisées plus loin.

---

## Étape 2 — Charger et préparer les données (Sentinel‑2, SRTM)

**Ce que nous faisons**

* On recentre la carte et on affiche l’AOI.
* On crée un **composite médian** Sentinel‑2 (période d’entraînement), masqué nuages.
* On calcule et **empile** les bandes/indices **prédicteurs** (features).
* On prépare la **pente** (SRTM) pour l’étape série temporelle.

**Code à coller (à la suite)**

```javascript
/* ===================== 2) DONNÉES ===================== */
Map.centerObject(AOI, 10);
Map.addLayer(AOI, {color: 'yellow'}, 'AOI');

// Sentinel-2 composite d’entraînement
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var s2Train = s2.filterBounds(AOI).filterDate(TRAIN_START, TRAIN_END).map(maskS2sr).median().clip(AOI);
s2Train = addIndices(s2Train).select(['B2','B3','B4','B8','B11','B12','NDWI','MNDWI','NDVI']);

// SRTM (pente)
var slope = ee.Terrain.slope(ee.Image('USGS/SRTMGL1_003')).rename('slope').clip(AOI);

// Empilement des prédicteurs
var featuresImg = s2Train
var PREDICTORS = ['B2','B3','B4','B8','B11','B12','NDWI','MNDWI','NDVI'];
print('Prédicteurs:', featuresImg.bandNames());
Map.addLayer(s2Train.select('NDWI'), {min:-1, max:1}, 'NDWI (train)');
```

**Ce que vous devez voir**

* La **zone AOI** en jaune.
* Une **couche NDWI (train)** en tons sombres/clairs (eau plus claire).
* Dans la **Console** : liste des noms de bandes prédictrices.

---

## Étape 3 — Auto‑étiquetage avec JRC GSW + échantillonnage

**Ce que nous faisons**

* On charge **GSW occurrence** et on fabrique deux masques :

  * **eau quasi‑certaine** (≥ 80 %)
  * **non‑eau quasi‑certaine** (≤ 5 %)
* On crée des **étiquettes 0/1** et on ne garde que la **zone sûre**.
* On fait un **échantillonnage stratifié équilibré** (autant de 0 que de 1) pour le ML.

**Analogie :** GSW joue le rôle de **prof** qui corrige les copies : on ne garde que les cas **très sûrs** pour expliquer au modèle ce qu’est « eau » vs « non‑eau ».

**Code à coller (à la suite)**

```javascript
/* ===================== 3) AUTO-ÉTIQUETAGE JRC GSW + ÉCHANTILLONNAGE ===================== */
// NB: si une version plus récente est dispo, adapter l’ID: 'JRC/GSW1_5/GlobalSurfaceWater'
var gsw = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').clip(AOI);

// Seuils “quasi certains”
var waterMask    = gsw.gte(GSW_WATER_MIN);
var nonWaterMask = gsw.lte(GSW_NONWATER_MAX);

// Image de labels 0/1
var labels = ee.Image(0).where(waterMask, 1).rename('label');

// “Zone sûre” = eau certaine OU non-eau certaine (léger érosion pour éviter bords)
var safe = waterMask.focal_min(1).unmask(0).or(nonWaterMask.focal_min(1).unmask(0)).selfMask();
Map.addLayer(waterMask.updateMask(waterMask), {palette:['blue']}, 'GSW eau (≥ seuil)', false);
Map.addLayer(nonWaterMask.updateMask(nonWaterMask), {palette:['red']}, 'GSW non-eau (≤ seuil)', false);

// Échantillonnage stratifié équilibré
var perClass = Math.floor(N_POINTS_TOTAL / 2);
var samples = featuresImg.addBands(labels)
  .updateMask(safe)
  .stratifiedSample({
    numPoints: 2000,          // ex. 4000 → 2000 si besoin
    classBand: 'label',
    classValues: [0, 1],
    classPoints: [perClass, perClass],
    region: AOI,
    scale: 10,
    seed: SEED,
    geometries: false                   // <<----- IMPORTANT
  });

print('Taille échantillons:', samples.size());
print('Histogramme étiquettes:', samples.aggregate_histogram('label'));
```

**Ce que vous devez voir**

* (En cochant les couches) les **zones eau** en **bleu** et **non‑eau** en **rouge** (optionnel).
* Dans la **Console** : la **taille** de l’échantillon et l’**histogramme des classes** (≈ moitié/moitié).

---

## Étape 4 — Split Train/Test, entraînement RF, évaluation

**Ce que nous faisons**

* On sépare **train/test** (70/30).
* On entraîne un **Random Forest** (200 arbres).
* On évalue sur le **test set** : matrice de confusion, accuracy, kappa.

**Code à coller (à la suite)**

```javascript
/* ===================== 4) SPLIT + ENTRAÎNEMENT RF + ÉVALUATION ===================== */
var withRandom = samples.randomColumn('rand', SEED);
var trainSet = withRandom.filter(ee.Filter.lt('rand', 0.7));
var testSet  = withRandom.filter(ee.Filter.gte('rand', 0.7));

// Contrôle que les deux classes existent des deux côtés
print('Train histogram:', trainSet.aggregate_histogram('label'));
print('Test  histogram:', testSet.aggregate_histogram('label'));

// Random Forest
var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  minLeafPopulation: 5,
  variablesPerSplit: null,
  bagFraction: 0.7,
  seed: 7
}).train({
  features: trainSet,
  classProperty: 'label',
  inputProperties: PREDICTORS
});

// Évaluation
var testClassified = testSet.classify(rf);
var confMatrix = testClassified.errorMatrix('label', 'classification');
print('Matrice de confusion', confMatrix);
print('Accuracy', confMatrix.accuracy());
print('Kappa', confMatrix.kappa());
```

**Ce que vous devez voir**
Dans la **Console** :

* la **matrice de confusion** (lignes = vérité, colonnes = prédiction),
* **Accuracy** (proportion bien classée) et **Kappa**.

---

## Étape 5 — Inférence : carte Eau/Non‑eau

**Ce que nous faisons**
On applique le modèle entraîné à l’image des **prédicteurs** pour afficher une **carte binaire** : `1 = eau`, `0 = non‑eau`.

**Code à coller (à la suite)**

```javascript
/* ===================== 5) INFÉRENCE : CARTE EAU/NON-EAU ===================== */
var classified = featuresImg.classify(rf).rename('water_class'); // 1=eau, 0=non-eau
var visRF = {min: 0, max: 1, palette: ['#d73027', '#1f78b4']};   // rouge=non-eau, bleu=eau
Map.addLayer(classified, visRF, 'RF eau/non-eau');
```

**Ce que vous devez voir**
Une **carte** : zones **bleues** = eau, **rouges** = non‑eau.

> Astuce lecture : zoomez près des **bords des plans d’eau** pour juger la qualité.

---

## Étape 6 — Série temporelle mensuelle (hectares)

**Ce que nous faisons**
On calcule, pour chaque mois de `TIMESERIES_START` à `TIMESERIES_END`, la **surface d’eau** (ha) dans l’AOI. Deux options :

* **A** : appliquer le **RF** (cohérent avec l’étape 5),
* **B** : utiliser **NDWI + seuil Otsu** (méthode non‑apprise).

**Analogie :** c’est comme **compter les cases bleues** mois par mois.

**Code à coller (à la suite)**

```javascript
/* ===================== 6) SÉRIE TEMPORELLE MENSUELLE (hectares) ===================== */
// Choix du masque d’eau mensuel : RF (true) ou NDWI+Otsu (false)
var USE_RF_FOR_TS = true;

var months = ee.List.sequence(
  0,
  ee.Date(TIMESERIES_END).difference(ee.Date(TIMESERIES_START), 'month').subtract(1)
);

function monthlyComposite(m) {
  m = ee.Number(m);
  var begin = ee.Date(TIMESERIES_START).advance(m, 'month');
  var stop  = begin.advance(1, 'month');

  var monthImg = s2.filterBounds(AOI).filterDate(begin, stop)
                   .map(maskS2sr).median().clip(AOI);
  monthImg = addIndices(monthImg)
              .select(['B2','B3','B4','B8','B11','B12','NDWI','MNDWI','NDVI'])
              .addBands(slope);

  // Option A: RF
  var waterMaskRF = monthImg.classify(rf).eq(1);

  // Option B: NDWI + Otsu
  var ndwi = monthImg.select('NDWI');
  var hist = ndwi.reduceRegion({
    reducer: ee.Reducer.histogram(255),
    geometry: AOI,
    scale: 30,
    bestEffort: true,
    tileScale: 4,
    maxPixels: 1e9
  }).get('NDWI');
  var thr = ee.Algorithms.If(hist, otsu(hist), 0.0);
  var waterMaskNDWI = ndwi.gte(ee.Number(thr));

  // >>> Masque final (corrigé)
  var waterMaskFinal = ee.Image(
    ee.Algorithms.If(USE_RF_FOR_TS, waterMaskRF, waterMaskNDWI)
  ).selfMask();

  var pixelArea = ee.Image.pixelArea().divide(10000); // ha
  var waterHa = pixelArea.updateMask(waterMaskFinal).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: AOI,
    scale: 10,
    bestEffort: true,
    tileScale: 4,
    maxPixels: 1e9
  }).get('area');

  return ee.Feature(null, {
    date: begin.format('YYYY-MM'),
    water_ha: waterHa
  });
}

var monthlyFeat = ee.FeatureCollection(months.map(monthlyComposite));
print('Time series (hectares) — premières lignes:', monthlyFeat.limit(5));

// Chart
var chart = ui.Chart.feature.byFeature({
  features: monthlyFeat, xProperty: 'date', yProperties: ['water_ha']
}).setOptions({
  title: 'Surface d’eau (ha) — mensuelle',
  hAxis: {title: 'Date', slantedText: true},
  vAxis: {title: 'Hectares'},
  lineWidth: 2, pointSize: 3
});
print(chart);
```

**Ce que vous devez voir**

* Dans la **Console** : un **tableau** des premières lignes `date / water_ha`.
* Un **graphique** montrant l’évolution mensuelle des hectares d’eau.

---

## Étape 7 — Exports (Drive)

**Ce que nous faisons**
On envoie :

* l’image binaire **RF_water_mask** (10 m) dans `GEE_Exports`,
* la table `water_area_monthly.csv` avec la série temporelle.

**Code à coller (à la suite)**

```javascript
/* ===================== 7) EXPORTS (Drive) ===================== */
if (DO_EXPORT_MAP) {
  Export.image.toDrive({
    image: classified,
    description: 'RF_water_mask',
    folder: 'GEE_Exports',
    fileNamePrefix: 'rf_water_mask',
    region: AOI,
    scale: 10,
    maxPixels: 1e13
  });
}

if (DO_EXPORT_TS) {
  Export.table.toDrive({
    collection: monthlyFeat,
    description: 'WaterArea_Timeseries',
    folder: 'GEE_Exports',
    fileNamePrefix: 'water_area_monthly',
    fileFormat: 'CSV'
  });
}
```

**Ce que vous devez voir**
Dans le **panneau Tasks** (en haut à droite), deux **tâches d’export** à lancer (`Run`).

---

## Étape 8 — Légende, importance des variables, métriques détaillées

**Ce que nous faisons**

* Ajouter une **légende** pour la carte RF.
* Visualiser l’**importance des variables** du RF (quelles bandes/indices comptent le plus ?).
* Afficher des **métriques détaillées** (accuracy globale, Kappa, Précision & Rappel par classe).

**Code à coller (à la suite)**

```javascript
// ===== Légende carte RF =====
function addLegend() {
  var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
  legend.add(ui.Label({
    value: 'Légende',
    style: {fontWeight: 'bold', fontSize: '12px'}
  }));

  var items = [
    {color: '#1f78b4', label: 'Eau (1)'},
    {color: '#d73027', label: 'Non-eau (0)'}
  ];

  items.forEach(function(it) {
    var colorBox = ui.Label('', {backgroundColor: it.color, padding: '8px', margin: '0 8px 4px 0'});
    var desc = ui.Label(it.label, {margin: '0 0 4px 0'});
    var row = ui.Panel([colorBox, desc], ui.Panel.Layout.Flow('horizontal'));
    legend.add(row);
  });

  Map.add(legend);
}
addLegend();

// ===== Importance des variables =====
var explain = rf.explain();
print('RF explain:', explain);
var importances = ee.Feature(null, explain).toDictionary().get('importance');

var keys = ee.Dictionary(importances).keys();
var values = ee.Dictionary(importances).values();

var fiTable = ee.FeatureCollection(ee.List.sequence(0, keys.size().subtract(1)).map(function(i) {
  i = ee.Number(i);
  return ee.Feature(null, {
    predictor: ee.List(keys).get(i),
    importance: ee.List(values).get(i)
  });
}));
print('Feature importance table', fiTable);

var fiChart = ui.Chart.feature.byFeature(fiTable, 'predictor', 'importance')
  .setChartType('ColumnChart')
  .setOptions({
    title: 'Importance des variables (RF)',
    hAxis: {title: 'Variables'},
    vAxis: {title: 'Importance (relative)'},
    legend: {position: 'none'}
  });
print(fiChart);

// ===== Matrice de confusion formatée =====
var cm = ee.ConfusionMatrix(confMatrix.array());
print('Matrice de confusion (format GEE):', cm);
print('Précision globale (Accuracy):', cm.accuracy());
print('Kappa:', cm.kappa());
print('Précision eau (User\'s Acc for classe 1):', cm.consumersAccuracy().get([1]));
print('Rappel eau (Producer\'s Acc for classe 1):', cm.producersAccuracy().get([1]));
```

**Ce que vous devez voir**

* Une **légende** en bas à gauche.
* Dans la **Console** :

  * un **tableau** d’importance des variables et un **diagramme en barres** ;
  * la **matrice de confusion** GEE et des **métriques** détaillées.

---

## Étape 9 — Conseils & variantes

* **Accuracy basse ?** Agrandir l’AOI, changer l’année `TRAIN_*`, augmenter `N_POINTS_TOTAL`, ajouter des **textures** (GLCM 3×3), vérifier visuellement nuages/ombres.
* **Classes déséquilibrées ?** Assouplir GSW (`eau ≥ 70`, `non‑eau ≤ 10`) ou élargir `AOI`.
* **Nuages/pluie persistants ?** Tester une **variante Sentinel‑1 (SAR)**.
* **Série temporelle :** jouer avec `USE_RF_FOR_TS` (`true`/`false`) pour comparer RF vs NDWI+Otsu.

---

## ✅ Résultats attendus

* Une **carte binaire Eau/Non‑eau** correcte (zones d’eau principales bien détectées).
* **Matrice de confusion** avec une accuracy raisonnable (dépend du site/période).
* Un **graphique** montrant la **variation mensuelle** de la surface d’eau (ha).
* Des **exports** (image GeoTIFF et CSV) dans **Google Drive › `GEE_Exports`**.

---

### (Optionnel) Pistes d’extension pour le cours

* Ajouter des **caractéristiques** : ratios supplémentaires, textures GLCM, distances aux rivières.
* Tester d’autres **modèles** (SVM, Gradient Tree Boost) et comparer.
* Construire une **évaluation spatiale** (train/test espacés dans l’aire) pour éviter la fuite spatiale.
* Appliquer le modèle à une **autre zone** (modifier `AOI`).
