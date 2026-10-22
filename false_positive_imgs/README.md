# False-positive QA samples

Twelve reviewed negative validation images incorrectly accepted as a passport or driving licence at confidence >= 0.20. Expected result for every image: none.

Images are grouped by source dataset and document design. See manifest.json for predicted labels, scores, box coordinates, image hashes and provenance. Original prepared JPEG bytes are preserved; boxes are metadata, not painted onto images.

These are validation-derived QA fixtures. Do not add them or other frames of these held-out designs to training. This selected regression set is not an unbiased benchmark. Russian internal passports are excluded from the accepted international passport class.

Reproduce with notebook 11's default negative gallery and export cell. Source dataset licensing and attribution requirements continue to apply.
