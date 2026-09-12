"""
Grad-CAM / Grad-CAM++ interpretability visualization.
ResNet-50 and Zoobot use standard GradCAM (both CNN-based).
ViT-Small uses GradCAM++ with a patch-token reshape transform.
Extracted from the project's main Colab notebook.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def reshape_transform_vit(tensor, height=14, width=14):
    """
    ViT outputs a sequence of patch tokens.
    Reshapes them back to 2D spatial format for GradCAM++.
    height=14, width=14 because 224/16 = 14 patches per side.
    """
    result = tensor[:, 1:, :]  # remove the class token
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def generate_gradcam(model, target_layer, model_name, device, class_names,
                      n_images=8, dataset=None, is_vit=False):
    """
    Generates and saves Grad-CAM (or Grad-CAM++ for ViT) visualizations,
    one example image per available class.
    """
    if is_vit:
        cam = GradCAMPlusPlus(
            model=model, target_layers=[target_layer],
            reshape_transform=reshape_transform_vit
        )
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])

    num_classes = len(class_names)
    sample_indices = []
    for class_id in range(min(n_images, num_classes)):
        for j in range(len(dataset)):
            _, lbl = dataset[j]
            if lbl == class_id:
                sample_indices.append(j)
                break
    while len(sample_indices) < n_images:
        sample_indices.append(len(sample_indices) * 50)

    fig, axes = plt.subplots(2, n_images, figsize=(20, 5))

    for plot_i, ds_idx in enumerate(sample_indices[:n_images]):
        img_tensor, true_label = dataset[ds_idx]
        input_t = img_tensor.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(input_t)
            pred_label = output.argmax(1).item()
            confidence = torch.softmax(output, 1).max().item()

        try:
            targets = [ClassifierOutputTarget(pred_label)]
            cam_map = cam(input_tensor=input_t, targets=targets)[0]
        except Exception as e:
            print(f"  CAM failed for image {plot_i}: {e}")
            cam_map = np.zeros((224, 224))

        img_show = img_tensor.permute(1, 2, 0).numpy()
        img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)
        img_show = img_show.astype(np.float32)

        vis = show_cam_on_image(img_show, cam_map, use_rgb=True)
        correct = "OK" if pred_label == true_label else "X"

        axes[0, plot_i].imshow(img_show)
        axes[0, plot_i].set_title(f"True:\n{class_names[true_label]}", fontsize=6)
        axes[0, plot_i].axis('off')

        axes[1, plot_i].imshow(vis)
        axes[1, plot_i].set_title(
            f"{correct} {class_names[pred_label]}\n({confidence*100:.0f}%)", fontsize=6
        )
        axes[1, plot_i].axis('off')

    plt.suptitle(f'Grad-CAM -- {model_name}', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'gradcam_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: gradcam_{model_name}.png")


def run_all_gradcam(resnet, vit, zoobot_model, test_ds, device, class_names):
    """
    Convenience wrapper: runs Grad-CAM for ResNet-50 and Zoobot,
    and Grad-CAM++ for ViT-Small, using the target layers from the paper.
    """
    resnet_target = resnet.layer4[-1]
    generate_gradcam(resnet, resnet_target, 'ResNet50', device, class_names,
                      dataset=test_ds, is_vit=False)

    vit_target = vit.blocks[-1].norm1
    generate_gradcam(vit, vit_target, 'ViT_Small', device, class_names,
                      dataset=test_ds, is_vit=True)

    zoobot_target = zoobot_model.encoder.blocks[-1]
    generate_gradcam(zoobot_model, zoobot_target, 'Zoobot', device, class_names,
                      dataset=test_ds, is_vit=False)
