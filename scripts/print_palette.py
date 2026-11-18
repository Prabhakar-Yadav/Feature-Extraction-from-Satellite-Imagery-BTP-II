
import re
for path in ['NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/hitl_storage/web/index.html', 'NEW_Model(sem2)/Phase_2_Model_Training/04_Multispectral_Subclass_Segmentation/scripts/hierarchical_pipeline.py']:
    try:
        content = open(path).read()
        print(path)
        for line in content.splitlines():
            if any(k in line.lower() for k in ['color', 'palette', 'class_names', 'classes']):
                print('  ', line[:100])
    except Exception as e:
        print(e)
