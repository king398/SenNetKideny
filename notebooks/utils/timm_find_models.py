import timm


models = timm.list_models(pretrained=True)
model_to_use = []
for i in models:
    if not timm.get_pretrained_cfg(i).fixed_input_size:
        try:
            if len(timm.create_model(i).feature_info) == 5:
                model_to_use.append(i)
        except:
            pass
