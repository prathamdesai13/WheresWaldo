import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by Antonio on 2018-01-03.
 */
public class ImageCropper {

    static final String path = "..";

    static int desiredWidth = 32, desiredHeight = 32;
    static int windowScale = 35;

    static File[] waldos = new File(path+"/Cropped Waldos/Waldos 35x35").listFiles(); //{"C:\\Users\\Antonio\\OneDrive - University of Waterloo\\Projects\\PycharmProjects\\WheresWaldo\\Maps\\4.png"};//
    static int waldo = -1; // Last Switched = 9_0_10.jpg

    static String outputPath = path+"/WaldoPicRepo/32/waldos/";

    public static void main(String[] args){
        ImageCropper ip = new ImageCropper();
        while (waldo < waldos.length){
            ip.switchWaldo();
            do {
                do {
                    ip.saveWaldo();
                } while (ip.moveImageRight());
                ip.moveImageToLeftSide();
            } while (ip.moveImageDown());
//            ip.centerImage();
//            ip.saveWaldo();
        }
    }

    int imageX = 0, imageY = 0;
    BufferedImage bi = null;

    public boolean moveImageUp(){
        imageY += windowScale;
        if (imageY > 0){
            moveImageToTop();
            return false;
        }
        return true;
    }
    public boolean moveImageDown(){
        imageY -= windowScale;
        if (imageY < -windowScale*(bi.getHeight()-desiredHeight)){
            moveImageToBottom();
            return false;
        }
        return true;
    }
    public boolean moveImageLeft(){
        imageX += windowScale;
        if (imageX > 0){
            moveImageToLeftSide();
            return false;
        }
        return true;
    }
    public boolean moveImageRight(){
        imageX -= windowScale;
        if (imageX < -windowScale*(bi.getWidth()-desiredWidth)){
            moveImageToRightSide();
            return false;
        }
        return true;
    }
    public void moveImageToTop(){
        imageY = 0;
    }
    public void moveImageToBottom(){
        imageY = -windowScale*(bi.getHeight()-desiredHeight);
    }
    public void moveImageToLeftSide(){
        imageX = 0;
    }
    public void moveImageToRightSide(){
        imageX = -windowScale*(bi.getWidth()-desiredWidth);
    }
    public void centerImage(){
        imageX = (desiredWidth-bi.getWidth())*windowScale/2;
        imageY = (desiredHeight-bi.getHeight())*windowScale/2;
    }
    int i = 0;
    public void saveWaldo(){
        try {
            String[] croppedWaldos = new File(outputPath).list();
            i++;
            File outputfile = new File(outputPath+i+".png");
            System.out.println(-imageX/windowScale+", "+-imageY/windowScale+", "+desiredWidth+", "+desiredHeight);
            ImageIO.write(bi.getSubimage(-imageX/windowScale, -imageY/windowScale, desiredWidth, desiredHeight), "png", outputfile);

        } catch (IOException e1) {
            e1.printStackTrace();
        }
    }
    public void switchWaldo(){
        ++waldo;
        if (waldo < waldos.length){
            try{
                System.out.println("Switching to:   "+waldos[waldo].getCanonicalPath()+"    ("+waldo+")");
                bi = ImageIO.read(new FileInputStream(waldos[waldo].getAbsolutePath()));//
                imageX = 0;
                imageY = 0;
                windowScale = 35;
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        else{
            System.out.println("All Waldos cropped!");
            System.exit(0);
        }
    }

}
