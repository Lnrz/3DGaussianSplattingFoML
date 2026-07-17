import data
import torch
from dataclasses import dataclass


@dataclass
class RenderContext:
    projections: data.ProjectedGaussians
    instances: data.GaussiansInstances
    tiles: data.ScreenTiles

    @classmethod
    def from_size(cls, gaussian_num: int, screensize=None, device=None):
        projections = data.ProjectedGaussians.from_size(gaussian_num, device)
        tiles = data.ScreenTiles.from_screensize(screensize, device) if screensize else None
        instances = data.GaussiansInstances.from_size(gaussian_num, device=device)

        return cls(projections, instances, tiles)
    
    def ensure_capacity(self, gaussian_num: int, screensize):
        self.projections.ensure_capacity(gaussian_num)
        self.instances.ensure_capacity(gaussian_num)
        self.tiles.ensure_capacity(screensize)
    
    def allocate_instances(self):
        self.instances.allocate_instances()


# TODO: since memory is reused attention should be taken when working with gaussians that need less memory than the allocated amount (ex. don't need to sort all keys, don't need to project all gaussians...)
def render(gs: data.Gaussians3D, cam: data.Camera, ctx: RenderContext=None):
    if isinstance(ctx, RenderContext):
        ctx.ensure_capacity(gs.num, cam.screensize)
    else:
        ctx = RenderContext.from_size(gs.num, cam.screensize)
    
    # project & filter
    torch.cumsum(ctx.instances.counts, dim=0, out=ctx.instances.offsets)
    ctx.allocate_instances()
    # create instances & keys
    torch.sort(ctx.instances.keys, dim=0, out=(ctx.instances.sorted_keys, ctx.instances.sorted_keys_indices))
    torch.index_select(ctx.instances.indices, dim=0, index=ctx.instances.sorted_keys_indices, out=ctx.instances.sorted_indices)
    # find tiles ranges
    # render & record final opacities
    # apply color offset if neeeded
    # return image & opacities (here too it's better to keep them for future uses instead of making new ones for every render, they depend on the screensize)