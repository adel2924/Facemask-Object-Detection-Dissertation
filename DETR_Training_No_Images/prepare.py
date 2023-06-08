import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

torch.set_grad_enabled(False);


# Get pretrained weights
checkpoint = torch.hub.load_state_dict_from_url(
            url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
            map_location='cpu',
            check_hash=True)

# Remove class weights
del checkpoint["model"]["class_embed.weight"]
del checkpoint["model"]["class_embed.bias"]

# Save
torch.save(checkpoint, r'/DETR/ahh/detr/detr-r50_no-class-head.pth')

first_class_index = 0


num_classes = 3

finetuned_classes = [
    'with_mask',
    'without_mask',
    'mask_weared_incorrect'
]


print('First class index: {}'.format(first_class_index))
print('Parameter num_classes: {}'.format(num_classes))
print('Fine-tuned classes: {}'.format(finetuned_classes))