"""Rebuild the Lewisville report from public aggregates and its existing screenshots.

    python scripts/build_lewisville_report.py

Requires pypdf, Pillow, and Edge or Chrome. The committed PDF supplies its eight
licensed screenshots; no new imagery or geometry is read. Public CSVs and SVGs
supply the report. Intermediate HTML and screenshots stay under ignored outputs/.
The original private report page is not required. A source PDF snapshot is retained
locally and the destination is replaced only after screenshot identity validation.
"""

import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/lewisville"
REPO = "https://github.com/efkopru/nearmap-building-footprints/blob/main/"
DEFAULT_PDF = RESULTS / "Lewisville_building_footprint_comparison.pdf"


def rows(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fmt(value):
    return f"{float(value):.2f}"


def number(value):
    return f"{int(value):,}"


def table(headers, data, css=""):
    return (f'<table class="{css}"><thead><tr>' + ''.join(f'<th>{escape(str(h))}</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row)
            + '</tr>' for row in data) + '</tbody></table>')


def chart(name):
    return '<div class="chart">' + (ROOT / f"docs/images/{name}.svg").read_text(encoding="utf-8") + '</div>'


def page(title, body):
    return f'<section class="report-page"><div class="eyebrow">Lewisville building footprint comparison</div><h1>{title}</h1>{body}</section>'


def link(name, label=None):
    return f'<a href="{REPO}results/lewisville/{name}">{escape(label or name)}</a>'


def build_html():
    reviewed = rows('reviewed_2026/test_area_scores.csv')
    reviewed_lookup = {(r['cohort'], r['method_key']): r for r in reviewed}
    order = ['esri_city', 'esri_local', 'sam3']
    reviewed_f1 = {cohort: ' / '.join(fmt(reviewed_lookup[cohort, method]['f1']) for method in order)
                   for cohort in ['all_sizes', 'at_least_20m2']}
    pages = []
    pages.append(page("SAM 3 and Esri's Mask R-CNN", '''
    <p class="lead">May 2026 Nearmap imagery, Lewisville, Texas. Independently reviewed test-area accuracy, with historical city-wide inventory agreement.</p>
    <p>Meta's SAM 3 and Esri's Building Footprint Extraction - USA model ran on the same city-wide aerial imagery at 10.16 cm per pixel. The frozen outputs were scored against independently reviewed test-area labels on 15 October 2026. Historical inventory comparisons and eight existing screenshots are retained separately.</p>
    <p><strong>Historical inventory:</strong> 38,924 source inventory features, dated primarily 2015 or earlier, with 33 undated. The city-wide evaluation contains 38,882 outlines. These are separate source and evaluated counts. Collection dates do not establish construction or demolition dates.</p>
    <p class="meta">29 September 2026 · reviewed test results added 15 October 2026</p>
    <div class="findings">
    <div class="finding"><b>REVIEWED_F1_MIN20</b><h2>Reviewed accuracy: ≥20 m²</h2><p>Test-area F1 for the city's Esri run, local Esri run and SAM 3, respectively, against 468 reviewed outlines. Both sets were filtered before matching.</p></div>
    <div class="finding"><b>REVIEWED_F1_ALL</b><h2>Reviewed accuracy: all sizes</h2><p>The same three frozen outputs against 600 reviewed outlines, including 132 under 20 m². This is a separate all-size evaluation.</p></div>
    <div class="finding"><b>0.83 vs 0.74</b><h2>Historical city-wide agreement</h2><p>The city's Esri run and SAM 3 F1 against inventory outlines of at least 20 m². These historical values are not reviewed city-wide accuracy.</p></div>
    <div class="finding"><b>4,923</b><h2>Review candidates</h2><p>SAM 3 outlines without an inventory match that also match Esri. Inventory omissions, geometry conventions, change and shared errors remain possible.</p></div>
    <div class="finding"><b>600 / 468</b><h2>Reviewed reference counts</h2><p>All-size and ≥20 m² reference outlines after AOI handling. Forty reference outlines crossing the AOI boundary were excluded.</p></div>
    <div class="finding"><b>Review complete</b><h2>Independent test labels</h2><p>Esad confirmed complete, blind test-area review on 15 October 2026, with no exceptions or unresolved cases. The receipt binds the labels and AOI used for scoring.</p></div>
    </div>
    <p class="note">An unmatched or missed outline is not dated evidence of construction, change or demolition.</p>
    ''' + '<p>Source details: ' + link('reviewed_2026/provenance.json', 'reviewed-reference provenance') + ', ' + link('reference_provenance.json', 'historical reference provenance') + ' and ' + link('scoring_audit.md', 'scoring audit') + '.</p>'))

    reviewed_table = table(['Area policy', 'Method', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'Median IoU'],
        [['All sizes' if cohort == 'all_sizes' else '≥20 m²', reviewed_lookup[cohort, method]['method'],
          *[reviewed_lookup[cohort, method][key] for key in ['tp', 'fp', 'fn']],
          *[fmt(reviewed_lookup[cohort, method][key]) for key in ['precision', 'recall', 'f1', 'median_iou']]]
         for cohort in ['all_sizes', 'at_least_20m2'] for method in order], 'compact')
    pages.append(page('Accuracy against reviewed 2026 test labels',
        '<p>Esad attested complete review of the 700 m test AOI against the May 2026 imagery, using only aerials and historical outlines with no prior or current prediction exposure. Review was confirmed on 15 October 2026, with no exceptions or unresolved cases. The six evaluations use the frozen outputs and a hash-bound review receipt.</p>'
        + chart('reviewed-test-accuracy') + reviewed_table
        + '<p>Forty reference outlines crossing the AOI boundary were excluded. The all-size reference has 600 outlines, including 132 below 20 m². For the ≥20 m² run, both sets were filtered after AOI handling and before one-to-one IoU ≥0.5 matching, leaving 468 reference outlines.</p>'
        + '<p class="note">Reviewed labels use an approximately 4 m² minimum convention; three smaller labels remain. Prediction cleanup uses a 4 m² minimum. The primary ≥20 m² result excludes those small labels.</p>'
        + '<p class="note">This result applies to the reviewed test square. It does not establish city-wide accuracy, a construction timeline or an improvement from retraining. The frozen settings were not re-tuned for these scores.</p>'
        + '<p>Evidence: ' + link('reviewed_2026/test_area_scores.csv', 'reviewed scores') + ' and ' + link('reviewed_2026/provenance.json', 'reviewed-reference provenance') + '.</p>'))

    screens = [
        ('screenshot-01.jpg', 'Typical residential area', '400 m square (33.0555° N, 96.9029° W). Both models outline almost every visible house. A few blue lines cross roofs where SAM 3 drew overlapping outlines.'),
        ('screenshot-02.jpg', 'Distribution centre', '400 m square (33.0558° N, 96.9157° W). Esri splits the largest roof at its chip edges. SAM 3 finds a corner and splits the smaller building where tiles meet.'),
        ('screenshot-03.jpg', 'Inventory-unmatched residential area', '300 m square (33.0255° N, 96.8972° W). Both models outline visible houses without inventory counterparts. This square has 125 shared outlines without an inventory match. No construction date is established.'),
        ('screenshot-04.jpg', "SAM 3's additional outlines", '300 m square (32.9923° N, 97.0167° W). A manufactured-home community beside a warehouse. The square has 57 SAM 3 outlines unmatched to both Esri and the inventory; they require review.'),
        ('screenshot-05.jpg', 'Where Esri finds more', '300 m square (33.0107° N, 96.9842° W). At an apartment complex, Esri outlines long carport and garage rows that SAM 3 mostly misses. The square has 30 inventory outlines found only by Esri.'),
        ('screenshot-06.jpg', 'Where SAM 3 finds more', '300 m square (33.0764° N, 97.0341° W). SAM 3 outlines more back-yard garages and sheds. The square has 24 inventory outlines found only by SAM 3; many small cyan outlines remain unmatched.'),
        ('screenshot-07.jpg', 'Small structures and open ground', '300 m square (33.0324° N, 96.9869° W). The square has 61 inventory outlines under 20 m² missed by both. Some cyan outlines overlap apparently open ground. This observation does not establish demolition or its date.'),
        ('screenshot-08.jpg', 'Large buildings', '300 m square (33.0680° N, 97.0150° W). Both models find long industrial roofs. SAM 3 splits them where tiles meet, and Esri edges step at chip boundaries.'),
    ]
    for first in (0, 4):
        gallery = '<div class="gallery">' + ''.join(f'<figure><img src="{file}"><figcaption><strong>{title}.</strong> {caption}</figcaption></figure>' for file, title, caption in screens[first:first+4]) + '</div>'
        intro = '<p>Red: the city\'s Esri run. Blue: SAM 3, "building", 1536 px, confidence threshold 0.7. Cyan: historical inventory.</p>'
        tail = '<p class="note">Eight existing screenshots: two typical areas and six 300 m squares selected by difference counts. The reviewer reported using only aerials and historical outlines for the test review, with no prior or current prediction exposure. Observations refer to May 2026 imagery, not verified dates of change.</p>'
        if first == 4:
            tail += '<p class="tip">Select a screenshot to see it larger.</p><p class="note">The imagery is licensed from Nearmap: keep this page private.</p>'
        pages.append(page('Screenshots: model agreement and differences' if first == 0 else 'Screenshots: missed structures and tile edges', intro + gallery + tail))

    city = rows('citywide_scores.csv')
    main = {r['method']: r for r in city if r['buildings'] == '20 m² and up'}
    city_table = table(['Method', 'Outlines', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'F1 ≥20 m²', 'Median IoU'],
        [[r['method'], number(int(r['matched']) + int(r['extra'])), number(r['matched']), number(r['extra']), number(r['missed']), fmt(r['precision']), fmt(r['recall']), fmt(r['f1']), fmt(main[r['method']]['f1']), fmt(r['median_iou'])] for r in city if r['buildings'] == 'all sizes'], 'compact')
    sizes = rows('citywide_by_size.csv')
    size_table = table(['Area', 'Inventory', 'Found by Esri', 'Found by SAM 3'], [[r['size'],number(r['outlines_2015']),number(r['found_by_esri_city']),number(r['found_by_sam3'])] for r in sizes])
    pages.append(page('Whole-city inventory agreement', '<p>All three runs used shared cleanup and 38,882 evaluated inventory outlines. SAM 3 used 6,538 tiles; the local Esri run used 150 chunks.</p>' + city_table + '<p class="note">The ≥20 m² column retains the historical published scores; see the scoring audit for execution policy. Other columns use all sizes. Extra outlines need independent review for omissions, conventions, registration and model errors.</p>' + chart('found-by-size') + size_table))

    outcome_table = table(['Area', 'Both', 'Only Esri', 'Only SAM 3', 'Neither'], [[r['size'],number(r['esri_city_and_sam3']),number(r['only_esri_city']),number(r['only_sam3']),number(r['missed_by_esri_city_and_sam3'])] for r in sizes])
    pages.append(page('Outcomes and model-agreement candidates', chart('outcomes-by-size') + outcome_table + '''
    <p>Both models found 24,585 inventory outlines; Esri alone found 1,917 and SAM 3 alone 1,173. Neither found 11,207, including 7,599 under 20 m². Adding local Esri, 10,463 were missed by all three, including 7,359 under 20 m².</p>
    <h2>Model-to-model agreement</h2><div class="stats"><div><b>29,567</b> matched pairs</div><div><b>91%</b> of Esri outlines</div><div><b>78%</b> of SAM 3 outlines</div><div><b>0.81</b> median IoU</div></div>
    <p>Matching uses IoU ≥ 0.5. Of SAM 3's 12,148 inventory-unmatched outlines, 4,923 match Esri. These are review candidates. An older inventory may omit structures or use a different geometry convention; both models may also make the same error.</p>
    <p class="note">The compatibility name <code>candidate_new</code> does not mean that a candidate is new. Do not use this layer to select or correct blind held-out annotations.</p>'''))

    old = rows('old_town_scores.csv')
    labels = {'sam3_text':'SAM 3', 'esri_city':'Esri, city run', 'esri_local':'Esri, local raw run'}
    old_table = table(['Method','TP','FP','FN','Precision','Recall','F1','Median IoU'],[[labels[r['label']],r['true_positives'],r['false_positives'],r['false_negatives'],fmt(r['precision']),fmt(r['recall']),fmt(r['f1']),fmt(r['matched_iou_median'])] for r in old])
    old_sizes = rows('old_town_by_size.csv')
    lookup = {(r['method'],r['size']):r for r in old_sizes}
    old_size_table = table(['Method','Recall ≥50 m²','Precision ≥50 m²','Recall <50 m²'],[[labels[k],fmt(lookup[k,'>=50 m2']['recall']),fmt(lookup[k,'>=50 m2']['precision']),fmt(lookup[k,'<50 m2']['recall'])] for k in labels])
    pages.append(page('Old Town: the untuned baseline','''<p>A 700 m square with 319 inventory outlines. Each method used default settings with the same cleanup and scoring. SAM 3 used 1024 px tiles and confidence threshold 0.5; local Esri used 0.5.</p>''' + old_table + old_size_table + '''<p>Per building: 109 found by all three methods, 38 by two, 32 by one, and 140 by none. Of the 140 missed by all, 96 are under 50 m². These per-building counts are retained from the original report.</p><h2>Interpretation boundary</h2><p>Differences between Old Town and the test AOI do not establish when structures were built, altered or demolished. Density, object size, inventory completeness, geometry convention, occlusion and model behaviour remain possible explanations.</p><p class="note">The baseline table uses all sizes. The tuning protocol requires filtering both sets to at least 20 m² before matching. Historical execution order must be checked separately.</p>'''))

    tuning = rows('tuning_final.csv')
    off = [r for r in tuning if r['regularize'] == 'off']
    lookup = {(r['method'],r['prompt'],r['tile_px'],r['threshold']):r for r in off}
    selected_keys = [('Esri city run','','',''),('SAM 3','building','2048','0.75'),('Esri local','','512','0.9'),('SAM 3','building','1536','0.7'),('SAM 3','house','1536','0.7'),('SAM 3','roof','2048','0.75'),('SAM 3','house','2048','0.5'),('SAM 3','building','1024','0.75'),('SAM 3','roof','1536','0.7'),('SAM 3','house','1024','0.4'),('SAM 3','roof','1024','0.8')]
    selected = [lookup[k] for k in selected_keys]
    def setting(r):
        if r['method'] == 'Esri city run':
            return 'as published (frozen)'
        if r['method'] == 'Esri local':
            return 'confidence ≥0.9 (frozen)'
        text = f"{r['prompt']}, {r['tile_px']} px, ≥{r['threshold']}"
        return text + (' (frozen)' if r['prompt']=='building' and r['tile_px']=='1536' and r['threshold']=='0.7' else '')
    bars = '<div class="bars">' + ''.join(f'<div class="barrow"><span>{escape(r["method"])}: {escape(setting(r))}</span><div><i style="width:{float(r["f1"])/.8*100:.3f}%;background:{"#cf4242" if r["method"] != "SAM 3" else "#2a78d6"}"></i><b>{fmt(r["f1"])}</b></div></div>' for r in selected) + '</div>'
    tuning_table = table(['Method','Setting','Precision','Recall','F1','Median IoU','Recall <20 m²'],[[r['method'],setting(r),fmt(r['precision']),fmt(r['recall']),fmt(r['f1']),fmt(r['median_iou']),fmt(r['recall_under_20_m2'])] for r in selected],'compact')
    pages.append(page('Tuning in Old Town','<p>The fixed objective was F1 at IoU ≥0.5 on buildings of at least 20 m². Before human review, tuning used the historical inventory under Amendment 1. All 176 variants are retained in ' + link('tuning_final.csv') + '. Regularization did not change a score; shown off.</p>' + bars + tuning_table))

    thresholds = ['0.3','0.4','0.5','0.6','0.7','0.75','0.8','0.85','0.9']
    heat = '<table class="heat"><thead><tr><th>Prompt</th><th>Tile size</th>' + ''.join(f'<th>≥{t}</th>' for t in thresholds) + '</tr></thead><tbody>'
    for prompt in ['building','house','roof']:
        for tile in ['1024','1536','2048']:
            heat += f'<tr><td>{prompt}</td><td>{tile} px</td>'
            for t in thresholds:
                val = float(lookup['SAM 3',prompt,tile,t]['f1'])
                lightness = 96 - (val / .65) * 55
                heat += f'<td style="background:hsl(213,75%,{lightness:.1f}%);color:{"white" if val>=.4 else "#14202a"}">{val:.2f}</td>'
            heat += '</tr>'
    heat += '</tbody></table>'
    pages.append(page('Confidence thresholds and the frozen rule', '<h2>SAM 3 F1 by prompt, tile size and confidence threshold</h2>' + heat + '''<p>Bigger tiles and higher confidence thresholds help up to about 0.75; above that, recall falls faster than precision rises. The best SAM 3 setting, 2048 px at 0.75 (F1 0.54), is within 0.01 of 1536 px at 0.7 (0.53). The rule selects the latter as closer to defaults.</p><p>The local Esri confidence sweep: ≥0.5: 0.43; ≥0.6: 0.46; ≥0.7: 0.49; ≥0.8: 0.51; ≥0.9: 0.53; ≥0.95: 0.48.</p><h2>Reproducing the area policy</h2><p><code>nbf evaluate --min-area-m2 20</code> removes predictions and reference polygons below 20 m² after AOI handling and before matching. This is the historical tuning protocol's policy.</p><p><code>--size-bins-m2 20 50 100</code> alone summarizes a global matching. Each object's own area determines its class; matched pairs may cross a bin boundary. Threshold summaries therefore need not equal a fresh filtered evaluation.</p><p>Keep the ≥20 m² and all-size runs separate, record the area filter and compare only reports with the same policy. The audit reproduces all six city/test ≥20 m² rows by retaining global matched pairs only when both objects meet 20 m². A before-matching filter rerun in the small test AOI gives the same counts. This does not prove historical operation order; the city and 176 tuning variants were not rerun with that filter.</p>''' + '<p>Evidence: ' + link('tuning_protocol.md', 'tuning protocol') + ' and ' + link('scoring_audit.md', 'scoring audit') + '.</p>'))

    test = rows('test_area_scores.csv')
    test_table = table(['Method','TP','FP','FN','Precision','Recall','F1','Median IoU','F1 all sizes'],[[r['method'],r['tp'],r['fp'],r['fn'],fmt(r['precision']),fmt(r['recall']),fmt(r['f1']),fmt(r['median_iou']),fmt(r['f1_all_sizes'])] for r in test],'compact')
    frozen = table(['Method','Frozen setting','Old Town F1'],[['Esri, city run','as published; regularization off','0.61'],['SAM 3','building; 1536 px; confidence ≥0.7; regularization off','0.53'],['Esri, local raw run','confidence ≥0.9; regularization off','0.53']])
    runtime = table(['Run','Area or tiles','Time'],[['SAM 3, 1024 px','90 tiles, Old Town','105 s; 0.82 s per tile'],['SAM 3, 1536 px','6,538 tiles, whole city','2 h 41 min; median 1.08 s model time/tile'],['Esri, local raw run','0.64 km², Old Town','4.9 min'],['Esri, local raw run','150 chunks, whole city','8 h 46 min model time']])
    pages.append(page('Frozen settings and held-out inventory scores','<p>A seeded random rule selected a 700 m test square. The frozen settings match the city-wide runs; the area was scored once against its inventory outlines.</p>' + frozen + '<h2>Historical scores reported for at least 20 m²</h2>' + test_table + '<p>The ≥20 m² columns retain historical published summaries; see the scoring audit for matching policy. The last column covers all sizes. Agreement exceeds Old Town; this difference does not establish its cause. At most 2 of the 136 inventory outlines under 20 m² were found.</p><h2>Runtime: laptop RTX 4050, 6 GB</h2>' + runtime + '<p>SAM 3 used up to 96% of GPU memory, stable through the city-wide run. Earlier extrapolations were about 3.3 h for 1024 px SAM 3 and 15 h for local Esri; the whole-city rows above are measurements.</p>'))

    pages.append(page('Methods, provenance and independent review', '''
    <dl><dt>Imagery</dt><dd>Nearmap mosaic, May 2026, 10.16 cm, 21.1 gigapixels, clipped to city limits. The GeoTIFF declared a local CRS in metres; EPSG:2276 in US survey feet was set through a VRT without resampling and checked against the city boundary.</dd>
    <dt>Historical reference</dt><dd>City of Lewisville inventory: 38,924 source features, primarily dated 2015 or earlier, 33 undated. City-wide evaluation: 38,882 outlines. The reported alignment of about 0.4 m does not establish completeness or temporal accuracy.</dd>
    <dt>Reviewed reference</dt><dd>May 2026 test labels: 640 total, 600 inside the AOI, 40 crossing excluded; 468 are at least 20 m². These are reviewed corrections of the inventory, retaining unchanged geometry and its attached-building splitting convention. No second reviewer or independent redraw is recorded.</dd>
    <dt>SAM 3</dt><dd>Text prompt through SamGeo 1.4.2 at a pinned commit. Tuning used 1024-2048 px tiles; city-wide inference used 1536 px with 192 px overlap, prompt "building", confidence threshold 0.7.</dd>
    <dt>Esri, city run</dt><dd>Building Footprint Extraction - USA (Mask R-CNN), ExtractFeaturesUsingAIModels defaults, including Esri post-processing. Published by the city.</dd>
    <dt>Esri, local raw</dt><dd>DetectObjectsUsingDeepLearning without NMS, retaining confidence scores. The whole city used 150 overlapping chunks of about 1.1 km, retaining outlines centred in each chunk's core.</dd>
    <dt>Scoring</dt><dd>EPSG:26914; shared prediction cleanup with a 4 m² minimum; one-to-one matching at IoU ≥0.5; AOI-crossing objects excluded. The protocol requires filtering both sets to at least 20 m² before matching; historical execution requires separate evidence.</dd></dl>
    <h2>Independent review and scoring</h2>
    <ol><li>Esad confirmed complete review of the test AOI against the May 2026 imagery on 15 October 2026, without viewing model outputs, agreement layers or candidates. No exceptions or unresolved cases were reported.</li><li>The frozen reviewed labels and AOI are bound to the receipt used by every evaluation. The receipt records the reviewer's attestations; software does not independently verify human review.</li><li>Six evaluations score three frozen outputs under separate all-size and ≥20 m² policies. No inference, tuning or retraining was performed for this accuracy run.</li></ol>
    <p><strong>Scope:</strong> accuracy is measured only in the reviewed test square. City-wide figures remain historical inventory agreement. Review uses visible roof edges of permanent buildings down to about 2 by 2 m, excluding open carports, canopies, pergolas, tents, trailers and vehicles. Fine-tuning remains unrun.</p>
    ''' + '<p>Public evidence: ' + ', '.join(link(n) for n in ['reviewed_2026/test_area_scores.csv','reviewed_2026/provenance.json','reference_provenance.json','scoring_audit.md','citywide_scores.csv','citywide_by_size.csv','test_area_scores.csv','old_town_scores.csv','old_town_by_size.csv','tuning_final.csv','tuning_protocol.md']) + '.</p>' + '<p class="note">Revised 15 October 2026 from the existing public report, its unchanged screenshots and committed result tables. The original private page was unavailable. The report is regenerated from public aggregate tables and its existing screenshots; it does not perform extraction, annotation or change detection.</p>'))

    css = '''
    @page { size: Letter; margin: 0.5in 0.5in 0.6in; }
    * { box-sizing: border-box; }
    body { font: 15px/1.48 "Segoe UI", Arial, sans-serif; color: #142427; background:white; margin:0; }
    .page { max-width: 920px; margin:0 auto; }
    .report-page { break-after:page; padding:0; }
    .report-page:last-child { break-after:auto; }
    .eyebrow { font-size:10px; color:#607176; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:7px; }
    h1 { font-size:26px; line-height:1.15; margin:0 0 15px; } h2 { font-size:17px; margin:19px 0 8px; }
    p { margin:10px 0; } .lead { font-size:18px; color:#28414a; } .meta,.note { font-size:12px; color:#53666c; }
    a { color:#236078; } code { font-size:12px; background:#eef3f3; padding:1px 3px; }
    .findings { display:grid; grid-template-columns:1fr 1fr; gap:13px; margin:18px 0; }
    .finding { border:1px solid #d7e1e2; border-radius:8px; padding:15px; }
    .finding > b { font-size:26px; color:#1f5367; } .finding h2 { margin:3px 0 8px; } .finding p { margin:0; font-size:14px; }
    .gallery { display:grid; grid-template-columns:1fr 1fr; gap:18px; max-width:none; }
    figure { margin:0; } figure img { width:100%; height:350px; display:block; object-fit:contain; }
    figcaption { font-size:12px; line-height:1.35; margin-top:7px; }
    table { width:100%; border-collapse:collapse; margin:15px 0; font-variant-numeric:tabular-nums; font-size:13px; }
    th,td { border-bottom:1px solid #d7e1e2; padding:9px 7px; text-align:right; vertical-align:top; }
    th { font-size:11px; font-weight:600; background:#eff4f4; color:#52686e; }
    th:first-child,td:first-child { text-align:left; } .compact { font-size:12px; } .compact th,.compact td { padding:8px 5px; }
    .chart { margin:14px 0; } .chart svg { display:block; width:100%; height:auto; }
    .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; } .stats div { font-size:12px; background:#eff4f4; padding:12px; } .stats b { display:block; font-size:24px; }
    .bars { margin:18px 0; } .barrow { display:grid; grid-template-columns:350px 1fr; align-items:center; gap:10px; margin:7px 0; font-size:12px; }
    .barrow > div { display:flex; align-items:center; gap:7px; height:17px; } .barrow i { height:17px; display:inline-block; border-radius:0 3px 3px 0; } .barrow b { font-size:12px; }
    .heat td { padding:16px 6px; font-weight:600; } .heat td:first-child { font-weight:400; } .heat th { padding:10px 6px; }
    dl { display:grid; grid-template-columns:140px 1fr; gap:12px 18px; margin:15px 0; font-size:13px; } dt { color:#52686e; } dd { margin:0; }
    li { margin:9px 0; } .tip { font-size:12px; }
    @media print { .report-page { break-inside:avoid; } .gallery { max-width:none !important; } }
    '''
    html = '<!doctype html><html><head><meta charset="utf-8"><title>Lewisville building footprint comparison</title><style>' + css + '</style></head><body><main class="page">' + ''.join(pages) + '</main></body></html>'
    return html.replace('REVIEWED_F1_MIN20', reviewed_f1['at_least_20m2']).replace('REVIEWED_F1_ALL', reviewed_f1['all_sizes'])


def screenshot_images(reader):
    """The report has exactly eight public 1000 px screenshot images."""
    found = [(index, image) for index, page in enumerate(reader.pages) for image in page.images]
    if len(found) != 8 or any(image.image.size != (1000, 1000) for _, image in found):
        raise ValueError("Expected exactly eight 1000 x 1000 screenshots in the report")
    return found


def screenshot_hashes(reader):
    return Counter(hashlib.sha256(image.data).hexdigest() for _, image in screenshot_images(reader))


def preserve_screenshots(source, rendered, destination):
    """Undo browser JPEG recompression and verify the unchanged screenshot set."""
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import NameObject

    original, current = PdfReader(source), PdfReader(rendered)
    writer = PdfWriter()
    # Clone before image decoding: pypdf image inspection can dereference colour
    # profiles in the reader, which otherwise leaves stale links in the clone.
    writer.clone_document_from_reader(current)
    originals, targets = screenshot_images(original), screenshot_images(current)
    for (_, image), (page_index, target) in zip(originals, targets):
        objects = writer.pages[page_index]["/Resources"]["/XObject"]
        key = NameObject("/" + target.name.rsplit(".", 1)[0])
        if key not in objects:
            raise ValueError("Screenshot must be a direct page image resource")
        # Clone the stream, including its colour-profile dependency. Cloning only
        # the indirect reference can leave an unresolved ICC object on rebuilds.
        objects[key] = image.indirect_reference.get_object().clone(writer).indirect_reference
    writer.compress_identical_objects()
    with destination.open("wb") as stream:
        writer.write(stream)
    checked = PdfReader(destination)
    if screenshot_hashes(original) != screenshot_hashes(checked):
        raise ValueError("Screenshot bytes changed during report generation")
    if [p.extract_text() for p in current.pages] != [p.extract_text() for p in checked.pages]:
        raise ValueError("Report text changed while preserving screenshots")
    return len(checked.pages)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-pdf", type=Path, default=DEFAULT_PDF,
                        help="Existing public report supplying the eight screenshots")
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "outputs/lewisville-report")
    parser.add_argument("--browser", help="Path to Edge or Chrome")
    parser.add_argument("--html-only", action="store_true", help="Prepare local source without rendering")
    args = parser.parse_args(argv)
    from pypdf import PdfReader
    import make_report_pdf

    source = args.source_pdf.resolve()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline = work / f"source-{source_hash}.pdf"
    if not baseline.exists():
        shutil.copyfile(source, baseline)
    if hashlib.sha256(baseline.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Preserved source PDF does not match its content hash")
    for index, (_, screenshot) in enumerate(screenshot_images(PdfReader(baseline)), 1):
        (work / f"screenshot-{index:02d}.jpg").write_bytes(screenshot.data)
    html = work / "report.html"
    html.write_text(build_html(), encoding="utf-8", newline="\n")
    if args.html_only:
        print(f"Wrote {html}")
        return
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed render or screenshot check cannot replace the existing report.
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        rendered, checked = Path(directory) / "rendered.pdf", Path(directory) / "checked.pdf"
        command = [str(html), str(rendered)]
        if args.browser:
            command += ["--browser", args.browser]
        make_report_pdf.main(command)
        pages = preserve_screenshots(baseline, rendered, checked)
        checked.replace(output)
    record = {"baseline_sha256": source_hash, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "pages": pages, "screenshots": 8, "screenshot_streams_identical_to_baseline": True,
              "source": "scripts/build_lewisville_report.py", "historical_scores_recomputed": False}
    (work / "artifact_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
