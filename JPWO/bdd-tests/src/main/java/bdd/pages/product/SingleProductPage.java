package bdd.pages.product;

import org.openqa.selenium.By;
import org.openqa.selenium.NoSuchElementException;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class SingleProductPage {

    private final WebDriver driver;
    private final WebDriverWait wait;

    private final By addToCartButton = By.cssSelector("[data-test='add-to-cart']");
    private final By addToFavoriteButton = By.cssSelector("[data-test='add-to-favorites']");
    private final By cartBadge = By.cssSelector("[data-test='cart-quantity']");
    private final By cartButton = By.cssSelector("[data-test='nav-cart']");
    private final By productName = By.cssSelector("[data-test='product-name']");
    private final By toastMessage = By.className("toast-message");

    public SingleProductPage(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public void addToCart() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(addToCartButton));
        button.click();
    }

    public void addToFavorites() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(addToFavoriteButton));
        button.click();
        wait.until(ExpectedConditions.visibilityOfElementLocated(toastMessage));
    }

    public void openCartPage() {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(cartButton));
        button.click();
    }

    public int getCartCount() {
        try {
            String countText = wait.until(
                    ExpectedConditions.visibilityOfElementLocated(cartBadge)
            ).getText();
            return Integer.parseInt(countText);
        } catch (NoSuchElementException e) {
            return 0;
        }
    }

    public String getProductName() {
        return wait.until(ExpectedConditions.visibilityOfElementLocated(productName)).getText();
    }
}
