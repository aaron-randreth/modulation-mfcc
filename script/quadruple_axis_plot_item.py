from typing import override

import pyqtgraph as pg

class QuadrupleAxisPlotItem(pg.PlotItem):
    central_row: int = 2
    column_count: int = 5
    row_count: int = 4


    def __init__(self) -> None:
        super().__init__()

        self.right = pg.ViewBox()
        self.right_bis = pg.ViewBox()

        self.left = self.vb
        self.left_bis = pg.ViewBox()

        self.right.setMouseEnabled(x=True, y=False)
        self.right_bis.setMouseEnabled(x=True, y=False)
        self.left.setMouseEnabled(x=True, y=False)
        self.left_bis.setMouseEnabled(x=True, y=False)

        self.shift_items_right()

        for axis_id in ("left", "bottom", "top"):
            self.axes[axis_id]["vb"] = self.vb

        self.setup_new_axes()
        self.position_axes()

        for axis in self.axes.values():
            axis["item"].hide()
            axis["items_count"] = 0

        self.getAxis("left").show()
        self.getAxis("bottom").show()

    def reset_layout_stretch(self) -> None:

        for i in range(self.row_count):
            self.layout.setRowPreferredHeight(i, 0)
            self.layout.setRowMinimumHeight(i, 0)
            self.layout.setRowSpacing(i, 0)
            self.layout.setRowStretchFactor(i, 1)
            
        for i in range(self.column_count):
            self.layout.setColumnPreferredWidth(i, 0)
            self.layout.setColumnMinimumWidth(i, 0)
            self.layout.setColumnSpacing(i, 0)
            self.layout.setColumnStretchFactor(i, 1)

        viewbox_row = 2
        self.layout.setRowStretchFactor(viewbox_row, 100)
        self.layout.setColumnStretchFactor(self.central_row, 100)

    def shift_items_right(self) -> None:
        # We know that all items only in pg.Plotitem only span 1 cell
        # We are shifting all items by one column to the right
        # so we go from right to left to avoid collisions.
        for col in reversed(range(self.layout.columnCount())):
            for row in range(self.layout.rowCount()):
                item = self.layout.itemAt(row, col)
                if item is None:
                    continue

                self.layout.removeItem(item)
                self.layout.addItem(item, row, col+1)

        for axis in self.axes.values():
            row, col = axis["pos"]
            axis["pos"] = (row, col+1)

    def add_viewboxes_to_scene(self) -> None:
        self.scene().addItem(self.right)
        self.scene().addItem(self.right_bis)
        self.scene().addItem(self.left_bis)

    def setup_new_axes(self) -> None:
        right_axis = self.getAxis("right")
        left_bis_axis = pg.AxisItem("left")
        right_bis_axis = pg.AxisItem("right")

        self.axes["right"]["vb"] = self.right
        self.axes["left_bis"] = {
            "item": left_bis_axis,
            "pos": (self.central_row, 0),
            "vb": self.left_bis
        }
        self.axes["right_bis"] = {
            "item": right_bis_axis,
            "pos": (self.central_row, self.column_count - 1),
            "vb": self.right_bis
        }

        right_axis.linkToView(self.right)
        left_bis_axis.linkToView(self.left_bis)
        right_bis_axis.linkToView(self.right_bis)

        self.right.setXLink(self)
        self.right_bis.setXLink(self)
        self.left_bis.setXLink(self)

        xmin, xmax = self.left.state["limits"]["xLimits"]
        self.right.setLimits(xMin=xmin, xMax=xmax)
        self.right_bis.setLimits(xMin=xmin, xMax=xmax)
        self.left_bis.setLimits(xMin=xmin, xMax=xmax)

        self.left.sigResized.connect(self.update_views)

    def position_axes(self) -> None:
        for axis in self.axes.values():
            self.layout.addItem(axis["item"], *axis["pos"])

    def update_views(self) -> None:
        self.right.setGeometry(self.left.sceneBoundingRect())
        self.right_bis.setGeometry(self.left.sceneBoundingRect())
        self.left_bis.setGeometry(self.left.sceneBoundingRect())

        self.right.linkedViewChanged(self.left, self.right.XAxis)
        self.right_bis.linkedViewChanged(self.left, self.right_bis.XAxis)
        self.left_bis.linkedViewChanged(self.left, self.left_bis.XAxis)

    def add_item(
        self,
        axis_id: str,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem
    ) -> None:
        if axis_id not in self.axes:
            raise ValueError(f"The axis {axis_id} does not exist.")

        axis = self.axes[axis_id]["item"]
        vb = self.axes[axis_id]["vb"]

        if not axis.isVisible():
            axis.show()

        vb.addItem(item)
        self.axes[axis_id]["items_count"] += 1

    def remove_item(
        self,
        axis_id: str,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem
    ) -> None:
        if axis_id not in self.axes:
            raise ValueError(f"The axis {axis_id} does not exist.")

        axis = self.axes[axis_id]["item"]
        vb = self.axes[axis_id]["vb"]
        items_count = self.axes[axis_id]["items_count"]

        if not axis.isVisible() or items_count == 0:
            raise ValueError(f"The chosen axis {axis_id} is empty.")

        vb.removeItem(item)
        items_count -= 1
        self.axes[axis_id]["items_count"] = items_count

        if items_count == 0:
            axis.hide()

# TODO Find a way for the dashboard to find their item when deleting
# 1) Store the item in dashboard ? Or,
# 2) Store the dashboard line id in the panel
class Panel(QuadrupleAxisPlotItem):
    item_count: int

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.item_count = 0

        self.rotation = {
            "left": None,
            "right": None,
            "left_bis": None,
            "right_bis": None,
        }

    def get_free_axis(self) -> str | None:
        for axis_id, axis_item in self.rotation.items():
            if axis_item is None:
                return axis_id

        return None

    def get_item_axis(
        self,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem
    ) -> str | None:
        for axis_id, axis_item in self.rotation.items():
            if axis_item is None:
                continue

            if axis_item is item:
                return axis_id

        return None

    def add_item(
        self,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem,
    ) -> None:
        if self.item_count >= 4:
            raise ValueError("This Panel already has 4 curves")

        self.item_count += 1

        axis_to_be_added_to = self.get_free_axis()
        if axis_to_be_added_to is None:
            raise ValueError("This Panel already has 4 curves")

        self.rotation[axis_to_be_added_to] = item
        super().add_item(axis_to_be_added_to, item)

    def remove_item(
        self,
        item: pg.PlotDataItem | pg.PlotCurveItem | pg.ScatterPlotItem,
    ) -> None:
        if self.item_count == 0:
            raise ValueError("This Panel does not have any curves")

        self.item_count -= 1
        axis_to_be_removed_from = self.get_item_axis(item)
        self.rotation[axis_to_be_removed_from] = None

        super().remove_item(axis_to_be_removed_from, item)
