package bdd.pages.cart;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;

public class CartPage {

    private final WebDriverWait wait;

    private final By productName = By.cssSelector("[data-test='product-title']");

    public CartPage(WebDriver driver) {
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    public boolean containsProduct(String expectedName) {
        WebElement nameElement = wait.until(ExpectedConditions.visibilityOfElementLocated(productName));
        return nameElement.getText().contains(expectedName);
    }
}
