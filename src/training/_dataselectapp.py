from pathlib import Path
import pickle
from datetime import datetime

from ipywidgets import widgets
from ipywidgets.widgets import widget_float
from ipywidgets.widgets.widget_layout import Layout

from IPython.display import display


from model_training.lib.keesds import show_images_button_grid
from model_training.lib.keesds import show_images, CSS_TABLE_STYLE

from siuba import _, mutate,  filter as _filter
import random 


def set_button_style(path, button, context):
    if path in context["save"]:
        button.button_style = "success"
    else:
        button.button_style = "warning"

 
def get_new_image(df, context, path_label, class_id):
    df = df[~df[path_label].isin(context["save"][class_id])]
    try:
        path = df[path_label].sample().iloc[0]
    except ValueError:
        return ""
    return path


def image_callback(df, context, class_id, _display: widgets.Text, path_label="image_path"):
    if class_id not in context["save"].keys():
        context["save"][class_id] = []
    def image_hook(img):
        def inner(button):
            # save courent image
            context["save"][class_id].append(button.tooltip)
            _display.value = str(len(set(context["save"][class_id])))
            # get new image to show
            path = get_new_image(df, context, path_label, class_id)
            set_button_style(path, button, context)
            button.tooltip = path
            # show new image
            try:
                image = open(path, "rb").read()
                img.value = image
            except FileNotFoundError:
                button.disabled = True
                img.value = b''
            
        return inner
    return image_hook

from ipywidgets import Widget

def select_app(logo_model, aspirant, context, save_path):

    def save_context():
        context['is_save'] = True
        pickle.dump(context, open(save_path + f'/context_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pickle', "wb"))

    # INIT THE APP!
    if context['is_save'] == False:
        context["save"] = {}
        context["unique_class_id"] = list(logo_model.meta["class_id"].unique())
    
    def gen_paths(df, class_id, n=5, replace = False, path_lable = "image_path"):
        return context["root_path_ex"] +"/"+ (df >> _filter(_.class_id == class_id)).sample(n, replace=replace)[path_lable]
    

    def new_class_id():
        random.shuffle(context["unique_class_id"])
        new_class_id = context["unique_class_id"].pop()
    
        tmp_df = (aspirant 
                   >> _filter(_.class_id_model == new_class_id)
                   >> mutate(image_path = context['root_path'] + _.image_path)
        
        )

        return new_class_id, tmp_df

    def _next(view):
        def inner(_):
            save_context()
            nonlocal view
            Widget.close_all() # <- need test if this is helping for memory consumption 
            class_id, tmp_df = new_class_id()

            view = render(class_id, tmp_df)
            display(view)
            
        return inner


    def render(class_id, tmp_df):
        context["tmp_viewed"] = []

        def update_df(df):
            try:
                return (df 
                        >> _filter(~_.image_path.isin(context['save'][class_id]))
                        >> _filter(~_.image_path.isin(context["tmp_viewed"])))
            except KeyError:
                return df

        # example
        ex_images = show_images(gen_paths(logo_model.meta, class_id, 4, True), 240, 260)

        # undo button
        undo = widgets.Button(description = "undo")
        # shuffle button
        shuffle = widgets.Button(description = "shuffle")
        # class_infos 
        class_info_names = ["class_id","logonummer", "variant", "name", "length",
                            "width", "height", "volume_bottles", "num_bottles", "num_images"]
        class_infos = widgets.HTML(CSS_TABLE_STYLE + 
                                    (logo_model.meta[class_info_names] 
                                     >> _filter(_.class_id == class_id)).iloc[0:1].to_html(index=False))
        # class_id_ info 
        id_info = widgets.Text(value = class_id,
                               layout = Layout(width="700px"))

        # info search images
        search_info = widgets.Text(value = str(tmp_df.shape[0]),
                                   layout = Layout(width="50px"))

        # click info 
        click_info = widgets.Text(value = "0",
                                   layout = Layout(width="50px"))

    
        # next button
        next = widgets.Button(description = "next")
        # grid 
        def gen_grid(context):
            image_paths = update_df(tmp_df)["image_path"]
            try:
                imgs = image_paths.sample(8)
            except ValueError:
                imgs = list(image_paths) + [""]* (8 - len(image_paths))
                context["tmp_viewed"] = []
            context["tmp_viewed"].extend(imgs)

            return show_images_button_grid(imgs, 4, 240, 262, image_callback(update_df(tmp_df), context, class_id, click_info))
        gridcon = gen_grid(context)
        # layout
        nav_bar = widgets.HBox([undo, shuffle, id_info, search_info, click_info, next])
        view = widgets.VBox([ex_images, class_infos, nav_bar, gridcon])

        # callbacks 
        def _shuffle(b):
            new_grid = gen_grid(context)
            old_grid = view.children[-1]
            view.children = view.children[:-1]
            del old_grid
            view.children += (new_grid,)

        def _undo(b):
            if len(context["save"][class_id]) > 0:
                context['save'][class_id].pop()
            click_info.value = str(len(set(context["save"][class_id])))
            
        shuffle.on_click(_shuffle)
        next.on_click(_next(view))
        undo.on_click(_undo)
        
        return view
    return render(*new_class_id())
